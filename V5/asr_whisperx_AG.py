import os
import time
import json
import gc
import warnings

import torch
import numpy as np
import whisperx
from whisperx.diarize import DiarizationPipeline
from transformers import pipeline
import whisper as openai_whisper
import soundfile as sf


def _select_best_cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    best_idx, best_free = 0, -1
    for idx in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
        except Exception:
            free_bytes = 0
        if free_bytes > best_free:
            best_free, best_idx = free_bytes, idx
    return best_idx


def _normalize_eos_token_id_for_whisper(model) -> None:
    """Normalize eos_token_id to int to avoid HF Whisper generation TypeError.

    Some fine-tuned Whisper checkpoints expose eos_token_id as a list in config.
    Transformers generation internals may slice with eos_token_id and crash when
    it is not an int: `TypeError: slice indices must be integers ...`.
    """
    for cfg_name in ("config", "generation_config"):
        cfg = getattr(model, cfg_name, None)
        if cfg is None:
            continue
        eos_token_id = getattr(cfg, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)) and eos_token_id:
            try:
                cfg.eos_token_id = int(eos_token_id[0])
            except Exception:
                pass


def transcribe_and_diarize(
    video_path: str,
    hf_token: str | None,
    output_dir: str = "output_asr",
    asr_backend: str = "transformers",  # "transformers" | "openai"
    asr_model_name: str | None = "distil-whisper/distil-large-v3.5",
    diarization_model: str | None = "pyannote/speaker-diarization-3.1",
    num_speakers: int | None = None,
    gpu_index: int | None = None,
    forced_language: str | None = None,
    skip_alignment: bool = False,  # Skip word-level alignment, keep only sentence-level
    asr_batch_size: int | None = None,
    asr_chunk_length_s: float = 30.0,
    asr_stride_length_s: float = 5.0,
):
    os.makedirs(output_dir, exist_ok=True)

    # Check if input is already a WAV file
    if video_path.lower().endswith(".wav"):
        wav_path = video_path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
    else:
        # Convert to WAV and save to wav/ subfolder
        from pydub import AudioSegment

        audio = AudioSegment.from_file(video_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        wav_dir = os.path.join(output_dir, "wav")
        os.makedirs(wav_dir, exist_ok=True)
        wav_path = os.path.join(wav_dir, f"{base_name}.wav")
        audio.export(wav_path, format="wav")

    gpu_idx = _select_best_cuda_device() if gpu_index is None else gpu_index
    device = f"cuda:{gpu_idx}"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot run on GPU.")
    if gpu_idx >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU index {gpu_idx} is not available. Only {torch.cuda.device_count()} GPU(s) available."
        )

    torch.cuda.set_device(gpu_idx)

    timings = {}

    # Load audio data once for all operations
    audio_data = whisperx.load_audio(wav_path)

    # --------------------
    # ASR
    # --------------------
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(gpu_idx)

    result = None
    selected_asr_name = None

    if asr_backend == "transformers":
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            model_name = asr_model_name or "distil-whisper/distil-large-v3.5"
            print(f"[ASR] backend=transformers model={model_name} forced_language={forced_language}")

            processor = AutoProcessor.from_pretrained(model_name)

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )
            model = model.to(device)
            _normalize_eos_token_id_for_whisper(model)

            first_param = next(model.parameters())
            if first_param.device.type != "cuda" or first_param.device.index != gpu_idx:
                raise RuntimeError(
                    f"Model failed to move to GPU {gpu_idx}. Current device: {first_param.device}"
                )

            asr = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=gpu_idx,
                torch_dtype=torch.float16,
            )

            gen_kwargs = {
                "num_beams": 1,
                "do_sample": False,
                "temperature": None,
                "return_dict_in_generate": False,
            }
            eos_token_id = getattr(model.generation_config, "eos_token_id", None)
            if isinstance(eos_token_id, (list, tuple)) and eos_token_id:
                eos_token_id = eos_token_id[0]
            if eos_token_id is None:
                eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                try:
                    gen_kwargs["eos_token_id"] = int(eos_token_id)
                except Exception:
                    pass
            if forced_language:
                gen_kwargs["language"] = forced_language

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                asr_call_kwargs = {
                    "return_timestamps": True,
                    "generate_kwargs": gen_kwargs,
                    "chunk_length_s": asr_chunk_length_s,
                    "stride_length_s": asr_stride_length_s,
                }
                if asr_batch_size is not None:
                    asr_call_kwargs["batch_size"] = int(asr_batch_size)
                asr_out = asr(wav_path, **asr_call_kwargs)

            chunks = asr_out.get("chunks") or []
            lang = asr_out.get("language", None)
            lang_prob = None

            result = {
                "language": lang,
                "language_probability": lang_prob,
                "segments": [
                    {
                        "start": (c.get("timestamp", [None, None])[0] or 0.0),
                        "end": (c.get("timestamp", [None, None])[1] or 0.0),
                        "text": (c.get("text", "").strip()),
                    }
                    for c in chunks
                    if c.get("text")
                ],
            }

            if not result["segments"]:
                result = {
                    "language": lang,
                    "language_probability": lang_prob,
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 0.0,
                            "text": asr_out.get("text", "").strip(),
                        }
                    ],
                }

            selected_asr_name = model_name

            del asr
            del model
            del processor
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ASR] transformers failed: {repr(e)}")
            raise

    elif asr_backend == "openai":
        whisper_model_name = asr_model_name or "small"
        print(f"[ASR] backend=openai model={whisper_model_name} forced_language={forced_language}")

        wmodel = openai_whisper.load_model(whisper_model_name, device=device)

        lang = None
        lang_prob = None

        if forced_language is None:
            try:
                import librosa

                loaded_audio = librosa.load(wav_path, sr=16000)[0]
                n_samples_30s = 30 * 16000
                mel = openai_whisper.log_mel_spectrogram(loaded_audio[:n_samples_30s]).to(device)
                _, probs = openai_whisper.detect_language(wmodel, mel)
                lang = max(probs, key=probs.get)
                lang_prob = float(probs[lang])
            except Exception:
                pass

        raw = wmodel.transcribe(
            wav_path,
            fp16=True,
            language=forced_language if forced_language else lang,
            beam_size=1,
            best_of=1,
            temperature=0,
        )

        detected_lang = raw.get("language", lang)
        result = {
            "language": detected_lang,
            "language_probability": lang_prob,
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"].strip(),
                }
                for s in raw.get("segments", [])
            ],
        }

        selected_asr_name = whisper_model_name

        del wmodel
        gc.collect()
        torch.cuda.empty_cache()

    else:
        raise ValueError(f"Unsupported asr_backend: {asr_backend}")

    timings["asr_s"] = time.time() - t0
    timings["asr_model"] = selected_asr_name
    torch.cuda.synchronize(gpu_idx)
    timings["asr_max_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated(gpu_idx))
    timings["asr_max_mem_alloc_gb"] = round(
        timings["asr_max_mem_alloc_bytes"] / (1024**3), 3
    )

    try:
        timings["asr_max_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(gpu_idx))
        timings["asr_max_mem_reserved_gb"] = round(
            timings["asr_max_mem_reserved_bytes"] / (1024**3), 3
        )
    except Exception:
        pass

    # --------------------
    # Alignment
    # --------------------
    if skip_alignment:
        timings["align_s"] = 0.0
        timings["align_max_mem_alloc_bytes"] = 0
        timings["align_max_mem_alloc_gb"] = 0.0
    else:
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats(gpu_idx)

        segments = result.get("segments", [])

        lang_for_alignment = result.get("language")
        if forced_language:
            lang_for_alignment = forced_language
        if not lang_for_alignment:
            lang_for_alignment = "en"

        if not segments:
            aligned_result = {"segments": []}
            result = aligned_result
        elif lang_for_alignment:
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=lang_for_alignment,
                    device=device,
                )
                aligned_result = whisperx.align(segments, model_a, metadata, audio_data, device)

                del model_a
                gc.collect()
                torch.cuda.empty_cache()

                result = aligned_result
            except Exception:
                pass

        timings["align_s"] = time.time() - t1
        torch.cuda.synchronize(gpu_idx)
        timings["align_max_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated(gpu_idx))
        timings["align_max_mem_alloc_gb"] = round(
            timings["align_max_mem_alloc_bytes"] / (1024**3), 3
        )

        try:
            timings["align_max_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(gpu_idx))
            timings["align_max_mem_reserved_gb"] = round(
                timings["align_max_mem_reserved_bytes"] / (1024**3), 3
            )
        except Exception:
            pass

    # --------------------
    # Diarization
    # --------------------
    t2 = time.time()
    torch.cuda.reset_peak_memory_stats(gpu_idx)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        try:
            dmodel_name = diarization_model or "pyannote/speaker-diarization-3.1"
            print(f"[DIAR] model={dmodel_name}")

            dmodel = DiarizationPipeline(
                model_name=dmodel_name,
                use_auth_token=hf_token,
                device=device,
            )
        except Exception as e:
            print(f"[DIAR] primary model failed: {repr(e)}")
            print("[DIAR] falling back to pyannote/speaker-diarization@2.1")

            dmodel = DiarizationPipeline(
                model_name="pyannote/speaker-diarization@2.1",
                use_auth_token=hf_token,
                device=device,
            )

        if num_speakers:
            dsegs = dmodel(
                audio_data,
                min_speakers=num_speakers,
                max_speakers=num_speakers,
            )
        else:
            dsegs = dmodel(audio_data)

        result = whisperx.assign_word_speakers(dsegs, result)

        del dmodel
        gc.collect()
        torch.cuda.empty_cache()

    timings["diar_s"] = time.time() - t2
    torch.cuda.synchronize(gpu_idx)
    timings["diar_max_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated(gpu_idx))
    timings["diar_max_mem_alloc_gb"] = round(
        timings["diar_max_mem_alloc_bytes"] / (1024**3), 3
    )

    try:
        timings["diar_max_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(gpu_idx))
        timings["diar_max_mem_reserved_gb"] = round(
            timings["diar_max_mem_reserved_bytes"] / (1024**3), 3
        )
    except Exception:
        pass

    # --------------------
    # Initialize detected_language fields
    # --------------------
    file_level_lang = result.get("language")
    for segment in result.get("segments", []):
        segment["detected_language"] = file_level_lang
        segment["detected_language_probability"] = None

    # --------------------
    # Per-segment language detection using OpenAI Whisper base
    # --------------------
    if result.get("segments"):
        try:
            lang_model = openai_whisper.load_model("base", device=device)

            segments = result.get("segments", [])
            sample_rate = 16000

            try:
                full_audio, sr = sf.read(wav_path)

                if sr != sample_rate:
                    import librosa

                    full_audio = librosa.resample(
                        full_audio,
                        orig_sr=sr,
                        target_sr=sample_rate,
                    )

                if full_audio.dtype != np.float32:
                    if full_audio.dtype == np.int16:
                        full_audio = full_audio.astype(np.float32) / 32768.0
                    elif full_audio.dtype == np.int32:
                        full_audio = full_audio.astype(np.float32) / 2147483648.0
                    else:
                        full_audio = full_audio.astype(np.float32)

                    if full_audio.max() > 1.0 or full_audio.min() < -1.0:
                        full_audio = full_audio / max(
                            abs(full_audio.max()),
                            abs(full_audio.min()),
                        )

            except Exception:
                try:
                    full_audio = np.array(audio_data, dtype=np.float32)
                    if full_audio.max() > 1.0 or full_audio.min() < -1.0:
                        full_audio = full_audio / max(
                            abs(full_audio.max()),
                            abs(full_audio.min()),
                        )
                except Exception:
                    full_audio = None

            if full_audio is not None:
                for segment in segments:
                    start_time = segment.get("start", 0.0)
                    end_time = segment.get("end", 0.0)

                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)

                    if end_sample > len(full_audio):
                        end_sample = len(full_audio)

                    if start_sample >= end_sample:
                        segment["detected_language"] = file_level_lang
                        segment["detected_language_probability"] = None
                        continue

                    segment_audio = full_audio[start_sample:end_sample].copy()

                    if len(segment_audio) < sample_rate * 0.3:
                        segment["detected_language"] = file_level_lang
                        segment["detected_language_probability"] = None
                        continue

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            if len(segment_audio.shape) > 1:
                                segment_audio = segment_audio.mean(axis=1)

                            segment_audio = segment_audio.astype(np.float32)

                            max_val = max(
                                abs(segment_audio.max()),
                                abs(segment_audio.min()),
                            )
                            if max_val > 1.0:
                                segment_audio = segment_audio / max_val

                            n_samples = 30 * sample_rate
                            if len(segment_audio) < n_samples:
                                padded = np.zeros(n_samples, dtype=np.float32)
                                padded[: len(segment_audio)] = segment_audio
                                segment_audio = padded
                            elif len(segment_audio) > n_samples:
                                start_idx = (len(segment_audio) - n_samples) // 2
                                segment_audio = segment_audio[start_idx : start_idx + n_samples]

                            mel = openai_whisper.log_mel_spectrogram(segment_audio).to(device)
                            _, probs = openai_whisper.detect_language(lang_model, mel)

                            detected_lang = max(probs, key=probs.get)
                            detected_prob = float(probs[detected_lang])

                            segment["detected_language"] = detected_lang
                            segment["detected_language_probability"] = round(detected_prob, 4)

                    except Exception:
                        segment["detected_language"] = file_level_lang
                        segment["detected_language_probability"] = None

            del lang_model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception:
            for segment in result.get("segments", []):
                if segment.get("detected_language") is None:
                    segment["detected_language"] = file_level_lang
                    segment["detected_language_probability"] = None

    # --------------------
    # Save to json/ subfolder
    # --------------------
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    out_json = os.path.join(json_dir, f"{base_name}_whisperx.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result, timings


if __name__ == "__main__":
    VIDEO_PATH = "/home/arfarh/DAVA/Conservatives_s1.mp4"
    HF_TOKEN = None
    ASR_BACKEND = "transformers"  # "transformers" | "openai"
    ASR_MODEL = "distil-whisper/distil-large-v3.5"
    NUM_SPEAKERS = None

    transcribe_and_diarize(
        video_path=VIDEO_PATH,
        hf_token=HF_TOKEN,
        output_dir="output_asr_Distil",
        asr_backend=ASR_BACKEND,
        asr_model_name=ASR_MODEL,
        diarization_model="pyannote/speaker-diarization-3.1",
        num_speakers=NUM_SPEAKERS,
        gpu_index=3,
    )
