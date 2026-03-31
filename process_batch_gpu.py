#!/usr/bin/env python3
"""
Batch processing script for ASR pipeline.
"""

import os
import sys
import json
import argparse
import subprocess
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Suppress logging from dependencies
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import torch
import numpy as np
import whisperx
import whisper as openai_whisper
from whisperx.diarize import DiarizationPipeline
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from pathlib import Path
from collections import Counter
import gc
import soundfile as sf

# Try to import tqdm for progress bar, fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None):
        if total:
            print(f"{desc}: 0/{total}", end="", flush=True)
        for i, item in enumerate(iterable, 1):
            if total:
                print(f"\r{desc}: {i}/{total}", end="", flush=True)
            yield item
        if total:
            print()


class BatchProcessor:
    """Processes multiple files in batches, keeping models loaded in GPU memory."""
    
    def __init__(self, gpu_index, asr_backend, asr_model, diarization_model, hf_token, forced_language=None, lang_detection_model_size="base", enable_topic_detection=False, topic_model_id="Qwen/Qwen3-0.6B", qwen_python_path=None, topic_verbose=False):
        self.gpu_index = gpu_index
        self.device = f"cuda:{gpu_index}"
        self.asr_backend = asr_backend
        self.asr_model_name = asr_model
        self.diarization_model = diarization_model
        self.hf_token = hf_token
        self.forced_language = forced_language
        self.lang_detection_model_size = lang_detection_model_size  # "tiny", "base", "small", "medium", "large"
        self.enable_topic_detection = enable_topic_detection
        self.topic_model_id = topic_model_id
        self.qwen_python_path = qwen_python_path or "/local/scratch/arfarh/Qwen/qwen3-env/bin/python3"
        self.topic_verbose = topic_verbose
        
        # Models (loaded once, reused for all files)
        self.asr_pipeline = None
        self.alignment_models = {}  # Cache alignment models by language
        self.diarization_pipeline = None
        self.lang_detection_model = None  # OpenAI Whisper model for language detection
        self.lang_detection_backend = "openai_whisper"  # Always using OpenAI Whisper
        # Topic detection uses external script with Qwen environment, no model loading here
        
        # Initialize GPU
        torch.cuda.set_device(gpu_index)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
    
    def load_asr_model(self):
        """Load ASR model once."""
        if self.asr_pipeline is not None:
            return
        
        if self.asr_backend == "transformers":
            model_name = self.asr_model_name or "distil-whisper/distil-large-v3.5"
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                dtype=torch.float16,
            )
            model = model.to(self.device)
            
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=self.gpu_index,
                dtype=torch.float16,
                ignore_warning=True,  # Suppress chunk_length_s warning
            )
        elif self.asr_backend == "openai":
            import whisper as openai_whisper
            free_bytes, _ = torch.cuda.mem_get_info(self.gpu_index)
            whisper_model_name = (
                "large-v2" if free_bytes > 18 * 1024**3 else ("medium" if free_bytes > 8 * 1024**3 else "small")
            )
            self.asr_pipeline = openai_whisper.load_model(whisper_model_name, device=self.device)
    
    def load_diarization_model(self):
        """Load diarization model once."""
        if self.diarization_pipeline is not None:
            return
        
        try:
            dmodel_name = self.diarization_model or "pyannote/speaker-diarization-3.1"
            self.diarization_pipeline = DiarizationPipeline(
                model_name=dmodel_name,
                use_auth_token=self.hf_token,
                device=self.device
            )
        except Exception:
            self.diarization_pipeline = DiarizationPipeline(
                model_name="pyannote/speaker-diarization@2.1",
                use_auth_token=self.hf_token,
                device=self.device
            )
    
    def load_language_detection_model(self):
        """Load OpenAI Whisper model for language detection."""
        if self.lang_detection_model is not None:
            return
        
        # Use OpenAI Whisper directly for language detection
        import whisper as openai_whisper
        model_size = self.lang_detection_model_size or "base"
        self.lang_detection_model = openai_whisper.load_model(model_size, device=self.device)
        self.lang_detection_backend = "openai_whisper"
    
    def get_alignment_model(self, language_code):
        """Get alignment model for language (cached)."""
        if language_code not in self.alignment_models:
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
            self.alignment_models[language_code] = (model_a, metadata)
        return self.alignment_models[language_code]
    
    def load_topic_model(self):
        """Check if topic detection script is available."""
        if not self.enable_topic_detection:
            return
        
        if not os.path.exists(self.qwen_python_path):
            self.enable_topic_detection = False
            return
        
        script_path = os.path.join(os.path.dirname(__file__), "qwen_topic_detector.py")
        if not os.path.exists(script_path):
            self.enable_topic_detection = False
            return
    
    def process_file(self, wav_path, out_dir, num_speakers=None, skip_alignment=False):
        """Process a single file using pre-loaded models."""
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        audio_data = whisperx.load_audio(wav_path)
        
        # ASR
        if self.asr_backend == "transformers":
            gen_kwargs = {
                "num_beams": 1,
                "do_sample": False,
                "temperature": None,
                "return_dict_in_generate": False,
            }
            if self.forced_language:
                gen_kwargs["language"] = self.forced_language
            
            # Suppress warnings during inference
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                asr_out = self.asr_pipeline(
                    wav_path,
                    return_timestamps=True,
                    generate_kwargs=gen_kwargs,
                    chunk_length_s=30,
                    stride_length_s=5,
                )
            chunks = asr_out.get("chunks") or []
            lang = asr_out.get("language", None)
            result = {
                "language": lang,
                "language_probability": None,
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
        elif self.asr_backend == "openai":
            raw = self.asr_pipeline.transcribe(
                wav_path,
                fp16=True,
                language=self.forced_language,
                beam_size=1,
                best_of=1,
                temperature=0,
            )
            detected_lang = raw.get("language", self.forced_language)
            result = {
                "language": detected_lang,
                "language_probability": None,
                "segments": [
                    {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                    for s in raw.get("segments", [])
                ],
            }
        
        # Alignment
        if not skip_alignment:
            lang_for_alignment = result.get("language") or self.forced_language or "en"
            segments = result.get("segments", [])
            if segments and lang_for_alignment:
                try:
                    model_a, metadata = self.get_alignment_model(lang_for_alignment)
                    aligned_result = whisperx.align(segments, model_a, metadata, audio_data, self.device)
                    result = aligned_result
                except Exception:
                    pass
        
        # Diarization
        if num_speakers:
            dsegs = self.diarization_pipeline(audio_data, min_speakers=num_speakers, max_speakers=num_speakers)
        else:
            dsegs = self.diarization_pipeline(audio_data)
        result = whisperx.assign_word_speakers(dsegs, result)
        
        # Initialize detected_language fields for all segments
        file_level_lang = result.get("language") or "en"
        
        for segment in result.get("segments", []):
            segment["detected_language"] = file_level_lang
            segment["detected_language_probability"] = None
        
        # Per-segment language detection using OpenAI Whisper medium (after ASR)
        if result.get("segments"):
            segments = result.get("segments", [])
            sample_rate = 16000
            
            # Load full audio for chunk extraction
            full_audio = None
            try:
                full_audio, sr = sf.read(wav_path)
                if sr != sample_rate:
                    import librosa
                    full_audio = librosa.resample(full_audio, orig_sr=sr, target_sr=sample_rate)
                # Normalize audio to float32 range [-1, 1]
                if full_audio.dtype != np.float32:
                    if full_audio.dtype == np.int16:
                        full_audio = full_audio.astype(np.float32) / 32768.0
                    elif full_audio.dtype == np.int32:
                        full_audio = full_audio.astype(np.float32) / 2147483648.0
                    else:
                        full_audio = full_audio.astype(np.float32)
                # Normalize to [-1, 1] range
                if full_audio.max() > 1.0 or full_audio.min() < -1.0:
                    max_val = max(abs(full_audio.max()), abs(full_audio.min()))
                    if max_val > 0:
                        full_audio = full_audio / max_val
            except Exception:
                # Fallback: use audio_data from whisperx
                try:
                    full_audio = np.array(audio_data, dtype=np.float32)
                    if full_audio.max() > 1.0 or full_audio.min() < -1.0:
                        max_val = max(abs(full_audio.max()), abs(full_audio.min()))
                        if max_val > 0:
                            full_audio = full_audio / max_val
                except Exception:
                    full_audio = None
            
            if full_audio is not None:
                try:
                    if self.lang_detection_model is None:
                        self.load_language_detection_model()
                    
                    for segment in segments:
                        start_time = segment.get("start", 0.0)
                        end_time = segment.get("end", 0.0)
                        
                        # Extract audio chunk for this segment
                        start_sample = int(start_time * sample_rate)
                        end_sample = int(end_time * sample_rate)
                        
                        # Ensure we don't go beyond audio length
                        if end_sample > len(full_audio):
                            end_sample = len(full_audio)
                        if start_sample >= end_sample:
                            segment["detected_language"] = file_level_lang
                            segment["detected_language_probability"] = None
                            continue
                        
                        segment_audio = full_audio[start_sample:end_sample].copy()
                        
                        # Skip if segment is too short (less than 0.3 seconds)
                        if len(segment_audio) < sample_rate * 0.3:
                            segment["detected_language"] = file_level_lang
                            segment["detected_language_probability"] = None
                            continue
                        
                        try:
                            # Use OpenAI Whisper for language detection
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                
                                # Ensure audio is mono and float32
                                if len(segment_audio.shape) > 1:
                                    segment_audio = segment_audio.mean(axis=1)
                                segment_audio = segment_audio.astype(np.float32)
                                
                                # Normalize to [-1, 1] if needed
                                max_val = max(abs(segment_audio.max()), abs(segment_audio.min()))
                                if max_val > 1.0:
                                    segment_audio = segment_audio / max_val
                                
                                # Detect language using OpenAI Whisper
                                # OpenAI Whisper requires 30 seconds, pad/truncate
                                n_samples = 30 * sample_rate  # 30 seconds at 16kHz = 480000
                                if len(segment_audio) < n_samples:
                                    # Pad with zeros to reach 30 seconds
                                    padded = np.zeros(n_samples, dtype=np.float32)
                                    padded[:len(segment_audio)] = segment_audio
                                    segment_audio = padded
                                elif len(segment_audio) > n_samples:
                                    # Truncate to 30 seconds (take middle portion for better representation)
                                    start_idx = (len(segment_audio) - n_samples) // 2
                                    segment_audio = segment_audio[start_idx:start_idx + n_samples]
                                
                                # Compute mel spectrogram
                                import whisper as openai_whisper
                                mel = openai_whisper.log_mel_spectrogram(segment_audio).to(self.device)
                                
                                # Detect language
                                _, probs = openai_whisper.detect_language(self.lang_detection_model, mel)
                                detected_lang = max(probs, key=probs.get)
                                detected_prob = float(probs[detected_lang])
                                
                                segment["detected_language"] = detected_lang
                                segment["detected_language_probability"] = round(detected_prob, 4)
                        except Exception:
                            segment["detected_language"] = file_level_lang
                            segment["detected_language_probability"] = None
                except Exception:
                    for segment in segments:
                        if segment.get("detected_language") is None:
                            segment["detected_language"] = file_level_lang
                            segment["detected_language_probability"] = None
            else:
                for segment in segments:
                    if segment.get("detected_language") is None:
                        segment["detected_language"] = file_level_lang
                        segment["detected_language_probability"] = None
        
        # Ensure all segments have language information
        for segment in result.get("segments", []):
            if segment.get("detected_language") is None:
                segment["detected_language"] = file_level_lang
                segment["detected_language_probability"] = None
        
        # Save JSON
        json_dir = os.path.join(out_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        out_json = os.path.join(json_dir, f"{base_name}_whisperx.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Run topic detection if enabled
        if self.enable_topic_detection:
            self.add_topics_to_json(out_json)
        
        return result
    
    def add_topics_to_json(self, json_path):
        """Add topic field to each segment in JSON file using external Qwen script."""
        if not self.enable_topic_detection:
            return
        
        # Check if topics already exist
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            segments = data.get("segments", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            if segments and all("topic" in seg for seg in segments):
                print(f"[TOPIC] Topics already exist in {os.path.basename(json_path)}, skipping")
                return
        except Exception:
            pass  # Continue with processing if we can't read the file
        
        try:
            # Call external topic detector script using Qwen Python environment
            script_path = os.path.join(os.path.dirname(__file__), "qwen_topic_detector.py")
            
            if not os.path.exists(script_path):
                print(f"[TOPIC] ERROR: Topic detector script not found at {script_path}")
                self._add_fallback_topics(json_path)
                return
            
            cmd = [
                self.qwen_python_path,
                script_path,
                "--json_file", json_path,
                "--model_id", self.topic_model_id,
                "--gpu_index", str(self.gpu_index),
            ]
            
            # Add verbose flag if enabled
            if self.topic_verbose:
                cmd.append("--verbose")
            
            print(f"[TOPIC] Running topic detection for {os.path.basename(json_path)}...")
            # Don't capture output if verbose mode - let it print to terminal
            if self.topic_verbose:
                result = subprocess.run(
                    cmd,
                    timeout=300  # 5 minute timeout per file
                )
                # Create a mock result object for compatibility
                class MockResult:
                    def __init__(self, returncode):
                        self.returncode = returncode
                    def __bool__(self):
                        return self.returncode == 0
                mock_result = MockResult(result.returncode)
                result = mock_result
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per file
                )
            
            if result.returncode == 0:
                # Verify that topics were actually added
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    segments = data.get("segments", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    segments_with_topic = sum(1 for seg in segments if "topic" in seg)
                    
                    if segments_with_topic == len(segments) and len(segments) > 0:
                        print(f"[TOPIC] ✓ Successfully added topics to {len(segments)} segments")
                    else:
                        print(f"[TOPIC] WARNING: Only {segments_with_topic}/{len(segments)} segments have topics, adding fallback")
                        self._add_fallback_topics(json_path)
                except Exception as e:
                    print(f"[TOPIC] ERROR: Failed to verify topics were added: {e}")
                    self._add_fallback_topics(json_path)
            else:
                print(f"[TOPIC] ERROR: Topic detection failed for {os.path.basename(json_path)} (exit code: {result.returncode})")
                if result.stdout:
                    print(f"[TOPIC] stdout: {result.stdout[:500]}")
                if result.stderr:
                    print(f"[TOPIC] stderr: {result.stderr[:500]}")
                self._add_fallback_topics(json_path)
                
        except subprocess.TimeoutExpired:
            print(f"[TOPIC] ERROR: Topic detection timed out for {os.path.basename(json_path)}")
            self._add_fallback_topics(json_path)
        except FileNotFoundError:
            print(f"[TOPIC] ERROR: Qwen Python not found at {self.qwen_python_path}")
            self._add_fallback_topics(json_path)
        except Exception as e:
            print(f"[TOPIC] ERROR processing topics for {os.path.basename(json_path)}: {e}")
            import traceback
            traceback.print_exc()
            self._add_fallback_topics(json_path)
    
    def _add_fallback_topics(self, json_path):
        """Add fallback 'other2' topic to all segments if topic detection fails."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            segments = data.get("segments", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            
            added_count = 0
            for segment in segments:
                if "topic" not in segment:
                    segment["topic"] = "other2"
                    added_count += 1
            
            if added_count > 0:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"[TOPIC] Added fallback 'other2' topic to {added_count} segments")
        except Exception as e:
            print(f"[TOPIC] ERROR: Failed to add fallback topics: {e}")
    
    def cleanup(self):
        """Clean up models."""
        if self.asr_pipeline:
            del self.asr_pipeline
            self.asr_pipeline = None
        if self.diarization_pipeline:
            del self.diarization_pipeline
            self.diarization_pipeline = None
        if self.lang_detection_model:
            del self.lang_detection_model
            self.lang_detection_model = None
        for model_a, _ in self.alignment_models.values():
            del model_a
        self.alignment_models.clear()
        gc.collect()
        torch.cuda.empty_cache()


def convert_to_wav(mp4_path, out_dir):
    """Convert MP4 to WAV."""
    mp4_dir = os.path.dirname(mp4_path)
    speaker_id = os.path.basename(mp4_dir)
    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
    
    speaker_wav_dir = os.path.join(out_dir, "wav", speaker_id)
    os.makedirs(speaker_wav_dir, exist_ok=True)
    wav_path = os.path.join(speaker_wav_dir, f"{base_name}.wav")
    
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", mp4_path, "-ac", "1", "-ar", "16000", "-y", wav_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        if os.path.exists(wav_path):
            return wav_path, speaker_id, base_name
    except subprocess.CalledProcessError:
        pass
    return None, None, None


def generate_speaker_json(csv_file, out_dir, lang, speaker_id):
    """Generate JSON file for a specific speaker."""
    mp4_to_speaker = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        for line in f:
            mp4_path = line.strip()
            if not mp4_path:
                continue
            if mp4_path.startswith("data/"):
                mp4_path = "/" + mp4_path
            mp4_dir = os.path.dirname(mp4_path)
            spk_id = os.path.basename(mp4_dir)
            base_name = os.path.splitext(os.path.basename(mp4_path))[0]
            mp4_to_speaker[base_name] = spk_id
    
    speaker_segments = []
    json_dir = os.path.join(out_dir, "json")
    
    for json_file in Path(json_dir).glob("*_whisperx.json"):
        base_name = json_file.stem.replace("_whisperx", "")
        spk_id = mp4_to_speaker.get(base_name, "UNKNOWN")
        
        if spk_id != speaker_id:
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                whisperx_data = json.load(f)
        except Exception:
            continue
        
        source_lang = whisperx_data.get("language", None)
        segments = whisperx_data.get("segments", [])
        
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            
            words = seg.get("words", [])
            if words and len(words) > 0:
                first_word = words[0]
                last_word = words[-1]
                start = first_word.get("start", seg.get("start", 0.0))
                end = last_word.get("end", seg.get("end", 0.0))
            else:
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
            
            duration = end - start
            speaker_number = seg.get("speaker")
            if not speaker_number:
                if words:
                    speakers = [w.get("speaker") for w in words if w.get("speaker")]
                    if speakers:
                        speaker_counter = Counter(speakers)
                        speaker_number = speaker_counter.most_common(1)[0][0]
                    else:
                        speaker_number = "UNKNOWN"
                else:
                    speaker_number = "UNKNOWN"
            
            wav_filename = f"{base_name}.wav"
            audio_filepath = f"wav/{speaker_id}/{wav_filename}"
            
            # Get detected language and confidence from segment (per-sentence language detection)
            detected_lang = seg.get("detected_language", source_lang)
            language_confidence = seg.get("detected_language_probability", None)
            
            segment_entry = {
                "audio_filepath": audio_filepath,
                "text": text,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "detected_language": detected_lang,
                "language_confidence": round(language_confidence, 4) if language_confidence is not None else None,
                "speakerID": speaker_id,
                "speakerNumber": speaker_number
            }
            
            if words and len(words) > 0:
                word_entries = []
                for word in words:
                    word_entry = {
                        "word": word.get("word", ""),
                        "start": round(word.get("start", 0.0), 3),
                        "end": round(word.get("end", 0.0), 3),
                        "score": round(word.get("score", 0.0), 3) if word.get("score") is not None else None,
                        "speaker": word.get("speaker", speaker_number)
                    }
                    word_entries.append(word_entry)
                segment_entry["words"] = word_entries
            
            speaker_segments.append(segment_entry)
    
    json_dir = os.path.join(out_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    
    if speaker_segments:
        json_file = os.path.join(json_dir, f"{speaker_id}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(speaker_segments, f, ensure_ascii=False, indent=2)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Batch processing for ASR pipeline on GPU")
    parser.add_argument("--csv_file", required=True, help="CSV file with MP4 paths")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of files to process before generating speaker JSON")
    parser.add_argument("--lang", default="", help="Language code (optional)")
    parser.add_argument("--hf_token", default="", help="HuggingFace token")
    parser.add_argument("--num_speakers", default="", help="Number of speakers")
    parser.add_argument("--gpu_index", type=int, default=6, help="GPU index")
    parser.add_argument("--asr_backend", default="transformers", help="ASR backend")
    parser.add_argument("--asr_model", default="distil-whisper/distil-large-v3.5", help="ASR model name")
    parser.add_argument("--skip_alignment", action="store_true", help="Skip word-level alignment")
    parser.add_argument("--lang_detection_model", default="base", help="Language detection model size: tiny, base, small, medium, large (default: base)")
    parser.add_argument("--enable_topic_detection", action="store_true", help="Enable topic detection using Qwen model")
    parser.add_argument("--topic_model_id", default="Qwen/Qwen3-0.6B", help="Qwen model ID for topic detection (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--qwen_python_path", default="/local/scratch/arfarh/Qwen/qwen3-env/bin/python3", help="Path to Qwen Python interpreter")
    parser.add_argument("--topic_verbose", action="store_true", help="Enable verbose output for topic detection (shows full prompt and response)")
    
    args = parser.parse_args()
    
    # Read CSV file
    mp4_files = []
    with open(args.csv_file, 'r', encoding='utf-8') as f:
        for line in f:
            mp4_path = line.strip()
            if not mp4_path:
                continue
            if mp4_path.startswith("data/"):
                mp4_path = "/" + mp4_path
            if os.path.exists(mp4_path):
                mp4_files.append(mp4_path)
    
    total_files = len(mp4_files)
    
    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "wav"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "json"), exist_ok=True)
    
    # Initialize batch processor (loads models once)
    processor = BatchProcessor(
        gpu_index=args.gpu_index,
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        diarization_model="pyannote/speaker-diarization-3.1",
        hf_token=args.hf_token if args.hf_token else None,
        forced_language=args.lang if args.lang else None,
        lang_detection_model_size=args.lang_detection_model,
        enable_topic_detection=args.enable_topic_detection,
        topic_model_id=args.topic_model_id,
        qwen_python_path=args.qwen_python_path,
        topic_verbose=args.topic_verbose
    )
    
    processor.load_asr_model()
    processor.load_diarization_model()
    processor.load_language_detection_model()
    if args.enable_topic_detection:
        processor.load_topic_model()
    
    # Process files in batches
    processed_speakers = set()
    num_speakers = int(args.num_speakers) if args.num_speakers else None
    
    # Create progress bar
    progress_bar = tqdm(
        enumerate(mp4_files, 1),
        total=total_files,
        desc="Processing files",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for i, mp4_path in progress_bar:
        # Update progress bar description with current file
        filename = os.path.basename(mp4_path)
        progress_bar.set_description(f"Processing: {filename[:40]}...")
        
        # Convert to WAV
        wav_path, speaker_id, base_name = convert_to_wav(mp4_path, args.out_dir)
        if wav_path is None:
            continue
        
        # Process file
        try:
            processor.process_file(
                wav_path=wav_path,
                out_dir=args.out_dir,
                num_speakers=num_speakers,
                skip_alignment=args.skip_alignment
            )
        except Exception as e:
            continue
        
        processed_speakers.add(speaker_id)
        
        # Generate speaker JSON after batch_size files or at the end
        if i % args.batch_size == 0 or i == total_files:
            progress_bar.set_description("Generating speaker JSON files...")
            for speaker_id in processed_speakers:
                generate_speaker_json(args.csv_file, args.out_dir, args.lang, speaker_id)
            processed_speakers.clear()
            progress_bar.set_description("Processing files")
    
    progress_bar.close()
    processor.cleanup()


if __name__ == "__main__":
    main()

