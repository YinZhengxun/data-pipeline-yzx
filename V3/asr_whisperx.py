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


def transcribe_and_diarize(
    video_path: str,
    hf_token: str | None,
    output_dir: str = "output_asr",
    asr_backend: str = "transformers",  # "transformers" | "openai" | "faster-whisper"
    asr_model_name: str | None = "distil-whisper/distil-large-v3.5",
    diarization_model: str | None = "pyannote/speaker-diarization-3.1",
    num_speakers: int | None = None,
    gpu_index: int | None = None,
    forced_language: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    from pydub import AudioSegment
    
    # Detect file format from extension
    file_ext = os.path.splitext(video_path)[1].lower().lstrip('.')
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"Converting audio to WAV (16kHz mono)...")
    print(f"  Input: {os.path.basename(video_path)} ({file_ext.upper()})")
    
    # Check if input is already a WAV file
    if file_ext == 'wav':
        # Check if it's already 16kHz mono
        try:
            existing_audio = AudioSegment.from_file(video_path)
            if existing_audio.frame_rate == 16000 and existing_audio.channels == 1:
                # Already in correct format, just copy or use directly
                wav_path = os.path.join(output_dir, f"{base_name}.wav")
                if video_path != wav_path:
                    import shutil
                    shutil.copy2(video_path, wav_path)
                    print(f"  ✓ WAV copied: {os.path.basename(wav_path)}")
                else:
                    print(f"  ✓ Using existing WAV: {os.path.basename(wav_path)}")
            else:
                # Need to convert
                audio = existing_audio.set_channels(1).set_frame_rate(16000)
                wav_path = os.path.join(output_dir, f"{base_name}.wav")
                audio.export(wav_path, format="wav")
                print(f"  ✓ WAV converted: {os.path.basename(wav_path)}")
        except:
            # If reading fails, try conversion
            audio = AudioSegment.from_file(video_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            wav_path = os.path.join(output_dir, f"{base_name}.wav")
            audio.export(wav_path, format="wav")
    else:
        # For non-WAV files, map extensions to pydub-compatible format names
        # Note: pydub doesn't support all formats directly, so we use a subset
        format_map = {
            'mp4': 'mp4',
            'webm': 'webm',
            'm4a': 'm4a',
            'mp3': 'mp3',
            'flac': 'flac',
            'ogg': 'ogg',
            'avi': 'avi',
            'flv': 'flv',
        }
        
        # For formats like mkv that pydub doesn't recognize, let it auto-detect
        audio_format = format_map.get(file_ext)
        if audio_format:
            try:
                audio = AudioSegment.from_file(video_path, format=audio_format)
            except:
                # If format fails, try auto-detection
                audio = AudioSegment.from_file(video_path)
        else:
            # Let pydub/ffmpeg auto-detect the format (works for mkv, etc.)
            audio = AudioSegment.from_file(video_path)
        
        # Convert to 16kHz mono WAV
        audio = audio.set_channels(1).set_frame_rate(16000)
        wav_path = os.path.join(output_dir, f"{base_name}.wav")
        audio.export(wav_path, format="wav")
        print(f"  ✓ WAV saved: {os.path.basename(wav_path)}")

    gpu_idx = _select_best_cuda_device() if gpu_index is None else gpu_index
    device = f"cuda:{gpu_idx}"
    torch.cuda.set_device(gpu_idx)
    print(f"Using CUDA device: {device}")

    timings = {}

    # Load audio data once for all operations
    audio_data = whisperx.load_audio(wav_path)

    # ASR
    print("\n=== ASR ===")
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(gpu_idx)
    
    result = None
    selected_asr_name = None
    
    if asr_backend == "transformers":
        try:
            print(f"Using Transformers ASR: {asr_model_name}")
            asr = pipeline(
                "automatic-speech-recognition",
                model=asr_model_name or "distil-whisper/distil-large-v3.5",
                device=gpu_idx,
            )
            gen_kwargs = {}
            if forced_language:
                gen_kwargs = {"language": forced_language}
            asr_out = asr(wav_path, return_timestamps=True, generate_kwargs=gen_kwargs)
            chunks = asr_out.get("chunks") or []
            result = {
                # Transformers pipeline may not expose language; do not force it
                "language": None,
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
                    "language": None,
                    "segments": [
                        {"start": 0.0, "end": 0.0, "text": asr_out.get("text", "").strip()}
                    ],
                }
            selected_asr_name = asr_model_name or "distil-whisper/distil-large-v3.5"
                
        except Exception as e:
            msg = str(e)
            print(f"  Transformers ASR failed: {msg}")
            # If user passed a faster-whisper repo (CT2 model.bin) or weights are missing for HF, try faster-whisper first
            if (asr_model_name and "faster-whisper" in asr_model_name) or (
                "does not appear to have a file named" in msg and "pytorch_model.bin" in msg
            ):
                print("  Switching to faster-whisper backend for CTranslate2 model...")
                asr_backend = "faster-whisper"
            else:
                print("  Falling back to OpenAI Whisper.")
                asr_backend = "openai"
                
    elif asr_backend == "faster-whisper":
        try:
            # Clear GPU cache before loading to avoid OOM
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Using faster-whisper ASR: {asr_model_name}")
            from faster_whisper import WhisperModel
            
            # More conservative compute types to avoid OOM
            # Allow overriding via env to mirror run_fasterWhisper.sh
            compute_type = os.environ.get("FAST_WHISPER_COMPUTE_TYPE", "int8_float16")
            beam_size = int(os.environ.get("FAST_WHISPER_BEAM", "1"))
            
            try:
                free_bytes, _ = torch.cuda.mem_get_info(gpu_idx)
                free_gb = free_bytes / (1024**3)
                print(f"Free VRAM before loading: {free_gb:.1f} GB")
            except Exception:
                pass
            
            print(f"Using compute_type: {compute_type}; beam_size: {beam_size}")
            
            # Try loading with the chosen compute type, fallback to int8 if cuDNN error
            compute_types_to_try = [compute_type]
            if compute_type != "int8":
                compute_types_to_try.append("int8")  # int8 often avoids cuDNN issues
            
            fw_model = None
            last_error = None
            for ct in compute_types_to_try:
                try:
                    print(f"Attempting with compute_type: {ct}")
                    fw_model = WhisperModel(
                        asr_model_name or "Systran/faster-whisper-large-v3",
                        device="cuda",
                        device_index=gpu_idx,
                        compute_type=ct,
                    )
                    compute_type = ct  # Update to the one that worked
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    if "cudnn" in error_str or "libcudnn" in error_str:
                        print(f"  cuDNN error with {ct}, trying fallback...")
                        if ct == "int8":
                            print(f"  cuDNN error even with int8. Try: pip install nvidia-cudnn-cu12")
                            raise RuntimeError(f"cuDNN error even with int8. Try: pip install nvidia-cudnn-cu12") from e
                        continue
                    else:
                        raise
            
            if fw_model is None:
                raise RuntimeError(f"Failed to load model: {last_error}")
            
            # Use smaller beam by default; can be overridden by env
            segments, info = fw_model.transcribe(
                wav_path,
                beam_size=beam_size,
                vad_filter=True,
                language=forced_language if forced_language else None,
            )
            segs = []
            for s in segments:
                segs.append({
                    "start": float(s.start) if s.start is not None else 0.0,
                    "end": float(s.end) if s.end is not None else 0.0,
                    "text": (s.text or "").strip(),
                })
            lang = getattr(info, "language", None) if info is not None else None
            result = {"language": lang, "segments": segs}
            selected_asr_name = f"{asr_model_name or 'Systran/faster-whisper-large-v3'} ({compute_type})"
            # Release model after transcription
            del fw_model
            fw_model = None
            gc.collect()
            torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  faster-whisper ASR failed: {e}. Falling back to OpenAI Whisper.")
            asr_backend = "openai"

    if asr_backend == "openai" and result is None:
        # Parse model name from asr_model_name if provided
        whisper_model_name = None
        if asr_model_name:
            # Extract model name from formats like "openai/whisper-large-v3" or just "large-v3"
            if "/" in asr_model_name:
                # Format: "openai/whisper-large-v3" -> "large-v3"
                parts = asr_model_name.split("/")
                if len(parts) > 1:
                    model_part = parts[-1]  # "whisper-large-v3"
                    if model_part.startswith("whisper-"):
                        whisper_model_name = model_part.replace("whisper-", "")  # "large-v3"
                    else:
                        whisper_model_name = model_part
            else:
                # Already just the model name like "large-v3"
                whisper_model_name = asr_model_name
        
        # If no model specified, select based on available GPU memory
        if not whisper_model_name:
            free_bytes, _ = torch.cuda.mem_get_info(gpu_idx)
            whisper_model_name = (
                "large-v2" if free_bytes > 18 * 1024**3 else ("medium" if free_bytes > 8 * 1024**3 else "small")
            )
            if whisper_model_name != "large-v2":
                print(f"Insufficient free VRAM ({free_bytes/1024**3:.1f} GB). Using '{whisper_model_name}'.")
        else:
            # Check if we have enough memory for the requested model
            free_bytes, _ = torch.cuda.mem_get_info(gpu_idx)
            if "large" in whisper_model_name and free_bytes < 18 * 1024**3:
                print(f"WARNING: Requested '{whisper_model_name}' but only {free_bytes/1024**3:.1f} GB free VRAM.")
                print(f"  This may cause out-of-memory errors. Consider using 'medium' or 'small'.")
        
        print(f"Using OpenAI Whisper ASR: {whisper_model_name}")
        wmodel = openai_whisper.load_model(whisper_model_name, device=device)
        raw = wmodel.transcribe(wav_path, fp16=True, language=forced_language if forced_language else None)
        result = {
            "language": raw.get("language", None),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in raw.get("segments", [])
            ],
        }
        selected_asr_name = whisper_model_name
    
    timings["asr_s"] = time.time() - t0
    timings["asr_model"] = selected_asr_name
    torch.cuda.synchronize(gpu_idx)
    timings["asr_max_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated(gpu_idx))
    timings["asr_max_mem_alloc_gb"] = round(timings["asr_max_mem_alloc_bytes"] / (1024**3), 3)
    try:
        timings["asr_max_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(gpu_idx))
        timings["asr_max_mem_reserved_gb"] = round(timings["asr_max_memory_reserved_bytes"] / (1024**3), 3)
    except Exception:
        pass
    

    print(f"ASR done in {timings['asr_s']:.2f}s using {selected_asr_name}")
    print(f"ASR peak GPU allocated: {timings.get('asr_max_mem_alloc_gb', 'n/a')} GB")

    # Alignment (forced alignment for word-level timestamps)
    print("\n=== Alignment ===")
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats(gpu_idx)
    segments = result.get("segments", [])
    # Determine language for alignment
    lang_for_alignment = result.get("language")
    if forced_language:
        lang_for_alignment = forced_language
    
    # Default to English if no language detected and none forced
    if not lang_for_alignment:
        print("WARNING: No language detected or specified. Defaulting to 'en' for alignment.")
        lang_for_alignment = "en"

    if not segments:
        aligned_result = {"segments": []}
        result = aligned_result
    elif lang_for_alignment:
        # Single-language alignment (forced alignment for word-level timestamps)
        try:
            model_a, metadata = whisperx.load_align_model(language_code=lang_for_alignment, device=device)
            aligned_result = whisperx.align(segments, model_a, metadata, audio_data, device)
            del model_a
            gc.collect()
            result = aligned_result
            print(f"Word-level alignment completed for language: {lang_for_alignment}")
        except Exception as e:
            print(f"WARNING: Alignment failed: {e}. Continuing without word-level timestamps.")
            # Continue with segment-level timestamps only
    else:
        # Should not reach here due to default above, but keep as safety
        print("WARNING: Alignment skipped - no language available")
    timings["align_s"] = time.time() - t1
    torch.cuda.synchronize(gpu_idx)
    timings["align_max_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated(gpu_idx))
    timings["align_max_mem_alloc_gb"] = round(timings["align_max_mem_alloc_bytes"] / (1024**3), 3)
    try:
        timings["align_max_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(gpu_idx))
        timings["align_max_mem_reserved_gb"] = round(timings["align_max_memory_reserved_bytes"] / (1024**3), 3)
    except Exception:
        pass
    print(f"Alignment done in {timings['align_s']:.2f}s")
    print(f"Alignment peak GPU allocated: {timings.get('align_max_mem_alloc_gb', 'n/a')} GB")

    # Diarization
    print("\n=== Diarization ===")
    t2 = time.time()
    torch.cuda.reset_peak_memory_stats(gpu_idx)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            dmodel_name = diarization_model or "pyannote/speaker-diarization-3.1"
            print(f"Loading diarization: {dmodel_name}")
            dmodel = DiarizationPipeline(model_name=dmodel_name, use_auth_token=hf_token, device=device)
        except Exception:
            print("Falling back to pyannote/speaker-diarization@2.1")
            dmodel = DiarizationPipeline(model_name="pyannote/speaker-diarization@2.1", use_auth_token=hf_token, device=device)
        if num_speakers:
            dsegs = dmodel(audio_data, min_speakers=num_speakers, max_speakers=num_speakers)
        else:
            dsegs = dmodel(audio_data)
        result = whisperx.assign_word_speakers(dsegs, result)
        del dmodel
        gc.collect()
    timings["diar_s"] = time.time() - t2
    torch.cuda.synchronize(gpu_idx)
    timings["diar_max_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated(gpu_idx))
    timings["diar_max_mem_alloc_gb"] = round(timings["diar_max_mem_alloc_bytes"] / (1024**3), 3)
    try:
        timings["diar_max_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(gpu_idx))
        timings["diar_max_mem_reserved_gb"] = round(timings["diar_max_memory_reserved_bytes"] / (1024**3), 3)
    except Exception:
        pass
    print(f"Diarization done in {timings['diar_s']:.2f}s")
    print(f"Diarization peak GPU allocated: {timings.get('diar_max_mem_alloc_gb', 'n/a')} GB")

    # Save
    out_json = os.path.join(output_dir, f"{base_name}_whisperx.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_json}")

    # Timing report
    timings_path = os.path.join(output_dir, f"{base_name}_timings.json")
    with open(timings_path, "w", encoding="utf-8") as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)
    print(f"Timing: {timings}")

    return result, timings


if __name__ == "__main__":
    VIDEO_PATH = "/home/arfarh/DAVA/Conservatives_s1.mp4"
    HF_TOKEN = None
    ASR_BACKEND = "transformers"  # "transformers" | "openai" | "faster-whisper"
    ASR_MODEL = "Systran/faster-whisper-large-v3"  # "distil-whisper/distil-large-v3.5" #"openai/whisper-large-v2" #"distil-whisper/distil-large-v3.5"  # used when ASR_BACKEND=transformers
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