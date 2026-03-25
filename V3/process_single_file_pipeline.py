#!/usr/bin/env python3
"""
Complete pipeline: ASR (WhisperX) → TXT → CoNLL-U → NER
Uses your existing asr_whisperx.py for transcription
"""

import os
import sys
import json
import argparse
import time
import datetime
from pathlib import Path

# Import your ASR function
from asr_whisperx import transcribe_and_diarize

try:
    import stanza
    import pandas as pd
    from transformers import pipeline
    try:
        import torch
    except Exception:
        torch = None

    # Optional topic modeling deps (lazy-checked later)
    try:
        from bertopic import BERTopic  # noqa: F401
        _has_bertopic = True
    except Exception:
        _has_bertopic = False
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        _has_st = True
    except Exception:
        _has_st = False
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install stanza pandas transformers torch")
    sys.exit(1)


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    seconds = max(0, int(round(seconds)))
    return str(datetime.timedelta(seconds=seconds))


def whisperx2txt(whisperx_result, txt_path):
    """Convert WhisperX JSON to TXT format with timestamps and speakers"""
    segments = whisperx_result.get("segments", [])
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        for seg in segments:
            start = seg.get("start", 0.0)
            speaker = seg.get("speaker") or "UNKNOWN"
            text = seg.get("text", "").strip()
            f.write(f"[{format_time(start)}] {speaker}: {text}\n")
    
    print(f"TXT saved: {txt_path}")


def whisperx2conllu(whisperx_result, conllu_path, file_basename, episode_filename, language="en"):
    """Convert WhisperX JSON to CoNLL-U format using Stanza"""
    try:
        import librosa
    except ImportError:
        librosa = None
    
    # Get audio duration if WAV exists
    audio_duration = 0.0
    wav_path = conllu_path.replace(".conllu", ".wav")
    if os.path.exists(wav_path) and librosa:
        try:
            audio_duration = librosa.get_duration(path=wav_path)
        except:
            pass
    # Fallback: calculate from last segment end time
    if audio_duration == 0.0:
        segments = whisperx_result.get("segments", [])
        if segments:
            audio_duration = segments[-1].get("end", 0.0)
    
    # Download Stanza model if needed
    print(f"Loading Stanza pipeline for {language}...")
    stanza.download(language, verbose=False)
    nlp = stanza.Pipeline(language, verbose=False)
    
    segments = whisperx_result.get("segments", [])
    
    # Filter segments with words
    texts = []
    valid_segments = []
    for seg in segments:
        words = seg.get("words", [])
        if len(words) > 0:
            texts.append(seg.get("text", "").strip())
            valid_segments.append(seg)
    
    if not texts:
        print(" No segments with words found!")
        return
    
    # Process with Stanza in batches
    print(f"Processing {len(texts)} segments with Stanza...")
    in_docs = [stanza.Document([], text=t) for t in texts]
    out_docs = []
    batch_size = 256
    for i in range(0, len(in_docs), batch_size):
        out_docs_batch = nlp(in_docs[i: i + batch_size])
        out_docs.extend(out_docs_batch)
    
    # Write CoNLL-U file
    print(f"Writing CoNLL-U format...")
    with open(conllu_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# newdoc id = {file_basename}\n")
        if episode_filename.endswith(".mp3"):
            f.write(f"# newdoc audio = {episode_filename}\n")
        else:
            f.write(f"# newdoc video = {episode_filename}\n")
        f.write(f"# newdoc start = 0.0\n# newdoc end = {audio_duration:.2f}\n\n")
        
        for sent_id, (doc, seg) in enumerate(zip(out_docs, valid_segments)):
            text = " ".join([s.text for s in doc.sentences])
            speaker = seg.get("speaker") or "UNKNOWN"
            start_time = seg.get("start", 0.0)
            end_time = seg.get("end", 0.0)
            
            # Sentence metadata
            f.write(f"# text = {text}\n")
            f.write(f"# sent_id = {sent_id}\n")
            f.write(f"# start = {start_time:.2f}\n")
            f.write(f"# end = {end_time:.2f}\n")
            f.write(f"# speaker = {speaker}\n")
            
            # Tokens - try to align WhisperX word timestamps with Stanza tokens
            whisperx_words = seg.get("words", [])
            
            # Simple alignment: use sentence timestamps, or try to match words
            for sentence in doc.sentences:
                word_idx = 0
                for word in sentence.words:
                    # Try to find matching WhisperX word for timestamp
                    word_time_start = start_time
                    word_time_end = end_time
                    
                    # Simple matching: check if word text matches any WhisperX word
                    for wx_word in whisperx_words:
                        wx_text = wx_word.get("word", "").strip().rstrip(".,!?;:")
                        if word.text.lower() == wx_text.lower():
                            word_time_start = wx_word.get("start", start_time)
                            word_time_end = wx_word.get("end", end_time)
                            break
                    
                    # CoNLL-U format: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
                    misc = f"start={word_time_start:.2f}|end={word_time_end:.2f}"
                    f.write(f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t_\t{word.head}\t{word.deprel}\t_\t{misc}\n")
                    word_idx += 1
            
            f.write("\n")
    
    print(f"✓ CoNLL-U saved: {conllu_path}")


def run_ner_on_conllu(conllu_path, ner_output_path, ner_model="flair", gpu_index=None):
    """Run NER on CoNLL-U sentences using Flair or HuggingFace transformers.

    Returns a dict with timing and (if CUDA) peak GPU memory metrics.
    """
    print(f"Running NER on {conllu_path} using {ner_model}...")
    
    # Check if CoNLL-U file exists
    if not os.path.exists(conllu_path):
        print(f"ERROR: CoNLL-U file not found: {conllu_path}")
        print("  Skipping NER (CoNLL-U file was not created)")
        return None
    
    # Set GPU device FIRST, before loading models
    cuda_metrics = {
        "ner_max_mem_alloc_bytes": None,
        "ner_max_mem_reserved_bytes": None,
    }
    current_device = None
    device_str = "cpu"
    
    if 'torch' in globals() and torch is not None and torch.cuda.is_available():
        try:
            if gpu_index is not None:
                torch.cuda.set_device(gpu_index)
                current_device = gpu_index
                device_str = f"cuda:{gpu_index}"
            else:
                current_device = torch.cuda.current_device()
                device_str = f"cuda:{current_device}"
            torch.cuda.reset_peak_memory_stats(current_device)
            print(f"Device set to use {device_str}")
        except Exception as e:
            print(f"WARNING: Could not set GPU device: {e}")
            current_device = None
            device_str = "cpu"
    
    # Load NER model with correct device
    if ner_model == "flair":
        try:
            from flair.data import Sentence
            from flair.models import SequenceTagger
            print("Loading Flair NER model (flair/ner-english-large)...")
            tagger = SequenceTagger.load('flair/ner-english-large')
            use_flair = True
        except ImportError:
            print("Flair not installed, falling back to HuggingFace")
            print("Install with: pip install flair")
            use_flair = False
            nlp_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device_str)
    else:
        use_flair = False
        nlp_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device_str)

    # Read sentences from CoNLL-U - extract from "# text = " lines
    sentences = []
    sent_ids = []
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# text = '):
                text = line.split('text = ', 1)[1].strip()
                sentences.append(text)
            elif line.startswith('# sent_id = '):
                sent_id = line.split('sent_id = ', 1)[1].strip()
                sent_ids.append(sent_id)
    
    # Extract entities
    all_entities = []
    t0 = time.time()
    for sent_id, sent in enumerate(sentences, 1):
        if use_flair:
            # Flair processing
            sentence = Sentence(sent)
            tagger.predict(sentence)
            entities = sentence.get_spans('ner')
            for span in entities:
                all_entities.append({
                    'sent_id': sent_id,
                    'entity': span.text,
                    'label': span.tag,  # PER, ORG, LOC, MISC
                    'score': round(float(span.score), 4),
                    'start': span.start_pos,
                    'end': span.end_pos
                })
        else:
            # HuggingFace processing
            entities = nlp_ner(sent)
            for ent in entities:
                all_entities.append({
                    'sent_id': sent_id,
                    'entity': ent.get('word', ''),
                    'label': ent.get('entity_group', ''),
                    'score': round(float(ent.get('score', 0.0)), 4),
                    'start': ent.get('start', 0),
                    'end': ent.get('end', 0)
                })
    ner_elapsed_s = time.time() - t0
    
    # Save as TSV
    if all_entities:
        import pandas as pd
        df = pd.DataFrame(all_entities)
        df.to_csv(ner_output_path, sep='\t', index=False)
        print(f"NER results saved: {ner_output_path} ({len(all_entities)} entities found)")
    else:
        print(f"No entities found")
    
    # Capture CUDA peak memory if available
    if 'torch' in globals() and torch is not None and torch.cuda.is_available() and current_device is not None:
        try:
            torch.cuda.synchronize(current_device)
            cuda_metrics["ner_max_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated(current_device))
            try:
                cuda_metrics["ner_max_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(current_device))
            except Exception:
                cuda_metrics["ner_max_mem_reserved_bytes"] = None
        except Exception:
            pass

    # Log timings and memory
    print(f"NER time: {ner_elapsed_s:.2f}s")
    if cuda_metrics["ner_max_mem_alloc_bytes"] is not None:
        alloc_gb = cuda_metrics["ner_max_mem_alloc_bytes"] / (1024**3)
        reserved_gb = (cuda_metrics["ner_max_mem_reserved_bytes"] or 0) / (1024**3)
        print(f"NER peak GPU allocated: {alloc_gb:.3f} GB; reserved: {reserved_gb:.3f} GB")

    return {
        "entities": all_entities,
        "ner_elapsed_s": round(ner_elapsed_s, 3),
        "ner_max_mem_alloc_bytes": cuda_metrics["ner_max_mem_alloc_bytes"],
        "ner_max_mem_reserved_bytes": cuda_metrics["ner_max_mem_reserved_bytes"],
    }


def run_emotion_recognition(segments, wav_path, result_json_path, gpu_index=None):
    """Run emotion2vec+ emotion recognition on each segment.
    
    segments: WhisperX-like list of {text, start, end, ...}
    wav_path: Path to 16kHz WAV file
    result_json_path: Path to WhisperX JSON file (will be updated with emotions)
    gpu_index: GPU index (optional)
    
    Returns dict with timing and emotion results.
    """
    import time
    t0 = time.time()
    
    # Check if emotion2vec dependencies are available
    try:
        from funasr import AutoModel
    except ImportError:
        print("Emotion recognition skipped: funasr is not installed.")
        print("Install with: pip install -U funasr modelscope")
        return None
    
    # Emotion labels mapping (from emotion2vec+ documentation)
    EMOTION_LABELS = {
        0: "angry",
        1: "disgusted", 
        2: "fearful",
        3: "happy",
        4: "neutral",
        5: "other",
        6: "sad",
        7: "surprised",
        8: "unknown"
    }
    
    print(f"Loading emotion2vec+ model (iic/emotion2vec_plus_large)...")
    try:
        # Load model (will auto-download on first use)
        model = AutoModel(model="iic/emotion2vec_plus_large")
        print("✓ Emotion model loaded")
    except Exception as e:
        print(f"ERROR: Could not load emotion model: {e}")
        return None
    
    # Check if WAV file exists
    if not os.path.exists(wav_path):
        print(f"WARNING: WAV file not found: {wav_path}")
        print("  Emotion recognition requires the audio file. Skipping...")
        return None
    
    # Load audio file
    try:
        import soundfile as sf
        import numpy as np
        audio_data, sample_rate = sf.read(wav_path)
        
        # Ensure 16kHz (emotion2vec requirement)
        if sample_rate != 16000:
            print(f"Resampling audio from {sample_rate}Hz to 16000Hz...")
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize to float32 range [-1, 1]
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Normalize amplitude
        max_val = max(abs(audio_data.max()), abs(audio_data.min()))
        if max_val > 0:
            audio_data = audio_data / max_val
            
    except Exception as e:
        print(f"ERROR: Could not load audio file: {e}")
        return None
    
    print(f"Processing {len(segments)} segments for emotion recognition...")
    
    # Process each segment
    emotion_results = []
    processed_count = 0
    
    for seg_idx, segment in enumerate(segments):
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        
        # Skip segments that are too short (< 0.1 seconds)
        if end_time - start_time < 0.1:
            segment["emotion"] = "unknown"
            segment["emotion_id"] = 8
            segment["emotion_score"] = None
            continue
        
        # Extract audio segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure we don't go beyond audio length
        if end_sample > len(audio_data):
            end_sample = len(audio_data)
        if start_sample >= end_sample:
            segment["emotion"] = "unknown"
            segment["emotion_id"] = 8
            segment["emotion_score"] = None
            continue
        
        segment_audio = audio_data[start_sample:end_sample].copy()
        
        # Save segment to temporary WAV file (emotion2vec needs file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_wav_path = tmp_file.name
            sf.write(tmp_wav_path, segment_audio, sample_rate)
        
        try:
            # Run emotion recognition using FunASR
            # Create temporary output directory
            temp_output_dir = tempfile.mkdtemp()
            
            try:
                res = model.generate(tmp_wav_path, output_dir=temp_output_dir, granularity="utterance", extract_embedding=False)
                
                # Parse result - FunASR may return dict or save to file
                emotion_id = None
                emotion_score = None
                
                # Check if result is a dict
                if isinstance(res, dict):
                    # Check for 'labels' and 'scores' (plural) - FunASR format
                    if "labels" in res and "scores" in res:
                        labels = res["labels"]
                        scores = res["scores"]
                        if isinstance(labels, list) and isinstance(scores, list) and len(labels) == len(scores):
                            # Find the label with highest score
                            max_idx = scores.index(max(scores))
                            label_str = labels[max_idx]
                            
                            # Parse label - could be "生气/angry" or just "angry" or index
                            # Map Chinese/English labels to emotion IDs
                            label_lower = label_str.lower()
                            if "angry" in label_lower or "生气" in label_str:
                                emotion_id = 0
                            elif "disgusted" in label_lower or "厌恶" in label_str:
                                emotion_id = 1
                            elif "fearful" in label_lower or "恐惧" in label_str:
                                emotion_id = 2
                            elif "happy" in label_lower or "开心" in label_str:
                                emotion_id = 3
                            elif "neutral" in label_lower or "中立" in label_str:
                                emotion_id = 4
                            elif "other" in label_lower or "其他" in label_str:
                                emotion_id = 5
                            elif "sad" in label_lower or "难过" in label_str:
                                emotion_id = 6
                            elif "surprised" in label_lower or "吃惊" in label_str:
                                emotion_id = 7
                            else:
                                # Use index if we can't parse
                                emotion_id = max_idx if max_idx < 8 else 4
                            
                            emotion_score = float(scores[max_idx])
                    
                    # Try other possible keys (fallback)
                    elif "label" in res:
                        emotion_id = res["label"]
                    elif "emotion" in res:
                        emotion_id = res["emotion"]
                    elif "prediction" in res:
                        emotion_id = res["prediction"]
                    elif "pred" in res:
                        emotion_id = res["pred"]
                    
                    if emotion_score is None:
                        if "score" in res:
                            emotion_score = res["score"]
                        elif "confidence" in res:
                            emotion_score = res["confidence"]
                        elif "prob" in res:
                            emotion_score = res["prob"]
                            
                elif isinstance(res, list) and len(res) > 0:
                    # If result is a list, take first element
                    first_res = res[0]
                    if isinstance(first_res, dict):
                        # Check for 'labels' and 'scores' format
                        if "labels" in first_res and "scores" in first_res:
                            labels = first_res["labels"]
                            scores = first_res["scores"]
                            if isinstance(labels, list) and isinstance(scores, list) and len(labels) == len(scores):
                                max_idx = scores.index(max(scores))
                                label_str = labels[max_idx]
                                
                                # Map label to emotion ID
                                label_lower = label_str.lower()
                                if "angry" in label_lower or "生气" in label_str:
                                    emotion_id = 0
                                elif "disgusted" in label_lower or "厌恶" in label_str:
                                    emotion_id = 1
                                elif "fearful" in label_lower or "恐惧" in label_str:
                                    emotion_id = 2
                                elif "happy" in label_lower or "开心" in label_str:
                                    emotion_id = 3
                                elif "neutral" in label_lower or "中立" in label_str:
                                    emotion_id = 4
                                elif "other" in label_lower or "其他" in label_str:
                                    emotion_id = 5
                                elif "sad" in label_lower or "难过" in label_str:
                                    emotion_id = 6
                                elif "surprised" in label_lower or "吃惊" in label_str:
                                    emotion_id = 7
                                else:
                                    emotion_id = max_idx if max_idx < 8 else 4
                                
                                emotion_score = float(scores[max_idx])
                        else:
                            # Fallback to other keys
                            emotion_id = first_res.get("label") or first_res.get("emotion") or first_res.get("prediction") or first_res.get("pred")
                            emotion_score = first_res.get("score") or first_res.get("confidence") or first_res.get("prob")
                elif isinstance(res, (int, float)):
                    # If result is directly an integer/float
                    emotion_id = int(res)
                
                # If still no result, try to read from output files
                if emotion_id is None:
                    # Check for output files in temp_output_dir
                    import glob
                    output_files = glob.glob(os.path.join(temp_output_dir, "*.npy")) + glob.glob(os.path.join(temp_output_dir, "*.txt"))
                    if output_files:
                        # Try to read from file (format may vary)
                        try:
                            # Try reading as text first
                            for out_file in output_files:
                                if out_file.endswith('.txt'):
                                    with open(out_file, 'r') as f:
                                        content = f.read().strip()
                                        # Try to parse emotion ID from content
                                        try:
                                            emotion_id = int(content)
                                            break
                                        except:
                                            pass
                        except:
                            pass
                
                # If we have emotion_id but no score, try to extract score from result
                if emotion_id is not None and emotion_score is None:
                    # Try to get score from the result structure
                    if isinstance(res, list) and len(res) > 0:
                        first_res = res[0]
                        if isinstance(first_res, dict) and "scores" in first_res:
                            scores = first_res["scores"]
                            if isinstance(scores, list) and len(scores) > emotion_id:
                                emotion_score = float(scores[emotion_id])
                
                # Convert to emotion name
                if emotion_id is not None:
                    emotion_id = int(emotion_id)
                    emotion_name = EMOTION_LABELS.get(emotion_id, "unknown")
                else:
                    # Default to neutral if we can't determine
                    emotion_id = 4
                    emotion_name = "neutral"
                
                # Add to segment
                segment["emotion"] = emotion_name
                segment["emotion_id"] = emotion_id
                if emotion_score is not None:
                    segment["emotion_score"] = float(emotion_score)
                else:
                    segment["emotion_score"] = None
                
                emotion_results.append({
                    "segment_idx": seg_idx,
                    "start": start_time,
                    "end": end_time,
                    "emotion": emotion_name,
                    "emotion_id": emotion_id,
                    "score": emotion_score
                })
                processed_count += 1
                
            finally:
                # Clean up temporary output directory
                try:
                    import shutil
                    shutil.rmtree(temp_output_dir)
                except:
                    pass
            
        except Exception as e:
            print(f"  WARNING: Failed to process segment {seg_idx} ({start_time:.2f}-{end_time:.2f}s): {e}")
            segment["emotion"] = "unknown"
            segment["emotion_id"] = 8
            segment["emotion_score"] = None
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_wav_path)
            except:
                pass
    
    # Update JSON file with emotion labels
    try:
        with open(result_json_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        result_data["segments"] = segments
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Updated JSON with emotion labels: {result_json_path}")
    except Exception as e:
        print(f"WARNING: Could not update JSON file: {e}")
    
    elapsed_s = time.time() - t0
    print(f"Emotion recognition time: {elapsed_s:.2f}s ({processed_count}/{len(segments)} segments processed)")
    
    # Count emotions
    emotion_counts = {}
    for seg in segments:
        emotion = seg.get("emotion", "unknown")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print(f"Emotion distribution: {emotion_counts}")
    
    return {
        "emotion_results": emotion_results,
        "emotion_elapsed_s": round(elapsed_s, 3),
        "emotion_counts": emotion_counts,
        "processed_count": processed_count
    }


def run_topic_modeling_from_segments(segments, topics_dir, base_name,
                                     embedding_model_name="all-MiniLM-L6-v2", gpu_index=None):
    """Run BERTopic over transcript segments and save results.

    segments: WhisperX-like list of {text, start, end, ...}
    topics_dir: directory to save outputs
    base_name: filename stem
    embedding_model_name: sentence-transformers model
    gpu_index: GPU index for tracking (optional)
    
    Returns dict with timing and GPU memory metrics.
    """
    # Initialize timing
    t0 = time.time()
    
    # Initialize GPU metrics
    cuda_metrics = {
        "topics_max_mem_alloc_bytes": None,
        "topics_max_mem_reserved_bytes": None,
    }
    current_device = None
    if 'torch' in globals() and torch is not None and torch.cuda.is_available():
        try:
            if gpu_index is not None:
                torch.cuda.set_device(gpu_index)
                current_device = gpu_index
            else:
                current_device = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(current_device)
        except Exception:
            current_device = None
    
    # Limit threads early to avoid resource exhaustion
    import os
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['NUMEXPR_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    
    os.makedirs(topics_dir, exist_ok=True)

    # Check dependencies
    try:
        from bertopic import BERTopic
    except Exception:
        print("Topic modeling skipped: BERTopic is not installed. pip install bertopic")
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        print("Topic modeling skipped: sentence-transformers is not installed. pip install sentence-transformers")
        return None

    # Prepare texts (use only non-empty lines)
    texts = [s.get("text", "").strip() for s in segments if s.get("text", "").strip()]
    if not texts:
        print("Topic modeling skipped: no text available.")
        return None

    # Limit to reasonable number of segments to avoid resource issues
    max_segments = 1000
    if len(texts) > max_segments:
        print(f"Topic modeling: Large dataset ({len(texts)} segments). Sampling {max_segments} segments for efficiency...")
        import random
        random.seed(42)
        texts = random.sample(texts, max_segments)

    print(f"Topic modeling: processing {len(texts)} segments...")
    print(f"Topic modeling: loading embedding model {embedding_model_name}...")
    
    # Load embedding model with limited threads
    try:
        embedder = SentenceTransformer(embedding_model_name, device='cpu')
        # Set batch size to avoid memory issues
        embedder._modules['0'].max_seq_length = 512
    except Exception as e:
        print(f"Warning: Could not set device/thread limits: {e}")
        embedder = SentenceTransformer(embedding_model_name)
    
    print("Topic modeling: fitting BERTopic (this may take a while)...")
    
    # Use simpler configuration to reduce resource usage
    try:
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Adjust min_df based on dataset size to avoid "no terms remain" error
        # For small datasets (< 100 segments), use min_df=1 or fraction
        # For larger datasets, use min_df=2
        if len(texts) < 100:
            # Small dataset: use fraction (0.01 = 1% of documents) or min_df=1
            min_df_val = max(1, int(len(texts) * 0.01))  # At least 1% of documents, but minimum 1
        else:
            min_df_val = 2
        
        # Use simpler components
        # Note: prediction_data=True is required if using transform() on batches
        umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, random_state=42, n_jobs=1)
        hdbscan_model = HDBSCAN(min_cluster_size=3, min_samples=1, prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=min_df_val, max_features=500)
        
        topic_model = BERTopic(
            embedding_model=embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=False,
            calculate_probabilities=False  # Disable to save memory
        )
    except Exception as e:
        # Fallback to default if components fail
        print(f"Using default BERTopic configuration (custom config failed: {e})...")
        topic_model = BERTopic(embedding_model=embedder, verbose=False, calculate_probabilities=False)
    
    # Process in smaller batches if needed
    # For large datasets, process all at once to avoid transform() issues
    try:
        if len(texts) > 1000:
            print(f"Topic modeling: Large dataset ({len(texts)} segments). Processing all at once...")
            topics, probs = topic_model.fit_transform(texts)
        elif len(texts) > 500:
            print("Topic modeling: processing in batches...")
            batch_size = 500
            all_topics = []
            all_probs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                if i == 0:
                    # Fit on first batch
                    batch_topics, batch_probs = topic_model.fit_transform(batch)
                else:
                    # Transform on subsequent batches (requires prediction_data=True in HDBSCAN)
                    try:
                        batch_topics, batch_probs = topic_model.transform(batch)
                    except AttributeError as e:
                        if "prediction_data" in str(e):
                            print(f"  WARNING: Batch transform failed, processing remaining {len(batch)} segments with fit_transform...")
                            # Fallback: fit_transform on remaining data
                            remaining_texts = texts[i:]
                            remaining_topics, remaining_probs = topic_model.fit_transform(remaining_texts)
                            all_topics.extend(remaining_topics)
                            all_probs.extend(remaining_probs if remaining_probs is not None else [None] * len(remaining_texts))
                            break
                        else:
                            raise
                all_topics.extend(batch_topics)
                all_probs.extend(batch_probs if batch_probs is not None else [None] * len(batch))
            topics = all_topics
            probs = all_probs if all_probs[0] is not None else None
        else:
            topics, probs = topic_model.fit_transform(texts)
    except ValueError as e:
        if "no terms remain" in str(e).lower() or "min_df" in str(e).lower():
            print(f"  WARNING: Vectorizer error ({e})")
            print("  Retrying with more lenient parameters (min_df=1, no stop words)...")
            # Retry with more lenient vectorizer
            try:
                from sklearn.feature_extraction.text import CountVectorizer
                lenient_vectorizer = CountVectorizer(min_df=1, max_features=500)  # Remove stop_words and min_df=2
                topic_model.vectorizer_model = lenient_vectorizer
                topics, probs = topic_model.fit_transform(texts)
                print("  ✓ Retry successful with lenient parameters")
            except Exception as retry_e:
                print(f"  ERROR: Retry also failed: {retry_e}")
                print("  Topic modeling failed - skipping this step")
                return None
        else:
            raise

    # Save topic info CSV
    import pandas as pd
    topic_info = topic_model.get_topic_info()
    csv_path = os.path.join(topics_dir, f"{base_name}_topics.csv")
    topic_info.to_csv(csv_path, index=False)
    print(f"Topics CSV saved: {csv_path}")

    # Try visualizations (optional, may require plotly)
    try:
        bar_fig = topic_model.visualize_barchart(top_n_topics=12)
        bar_html = os.path.join(topics_dir, f"{base_name}_topics_barchart.html")
        bar_fig.write_html(bar_html)
        print(f"Topics barchart saved: {bar_html}")
    except Exception:
        pass

    try:
        hier_fig = topic_model.visualize_hierarchy()
        hier_html = os.path.join(topics_dir, f"{base_name}_topics_hierarchy.html")
        hier_fig.write_html(hier_html)
        print(f"Topics hierarchy saved: {hier_html}")
    except Exception:
        pass

    # Calculate elapsed time
    topics_elapsed_s = time.time() - t0
    
    # Capture CUDA peak memory if available
    if 'torch' in globals() and torch is not None and torch.cuda.is_available() and current_device is not None:
        try:
            torch.cuda.synchronize(current_device)
            cuda_metrics["topics_max_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated(current_device))
            try:
                cuda_metrics["topics_max_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(current_device))
            except Exception:
                cuda_metrics["topics_max_mem_reserved_bytes"] = None
        except Exception:
            pass
    
    # Print timing and memory info
    print(f"Topic modeling time: {topics_elapsed_s:.2f}s")
    if cuda_metrics["topics_max_mem_alloc_bytes"] is not None:
        alloc_gb = cuda_metrics["topics_max_mem_alloc_bytes"] / (1024**3)
        reserved_gb = (cuda_metrics["topics_max_mem_reserved_bytes"] or 0) / (1024**3)
        print(f"Topic modeling peak GPU allocated: {alloc_gb:.3f} GB; reserved: {reserved_gb:.3f} GB")
    
    return {
        "topics": topics,
        "probs": probs,
        "info_path": csv_path,
        "topics_elapsed_s": round(topics_elapsed_s, 3),
        "topics_max_mem_alloc_bytes": cuda_metrics["topics_max_mem_alloc_bytes"],
        "topics_max_mem_reserved_bytes": cuda_metrics["topics_max_mem_reserved_bytes"],
    }


def main():
    parser = argparse.ArgumentParser(description="Complete pipeline: ASR → TXT → CoNLL-U → NER")
    parser.add_argument("--episode_path", required=True, help="Path to video/audio file")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for diarization")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers")
    parser.add_argument("--out_dir", default="output_pipeline", help="Output directory")
    parser.add_argument("--asr_backend", default="transformers", choices=["transformers", "openai", "faster-whisper"])
    parser.add_argument("--asr_model", default="distil-whisper/distil-large-v3.5", help="ASR model name") #distil-whisper/distil-large-v3.5
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index")
    parser.add_argument("--skip_ner", action="store_true", help="Skip NER step")
    parser.add_argument("--ner_model", default="flair", choices=["flair", "bert"], 
                       help="NER model: 'flair' (more accurate, slower) or 'bert' (faster)")
    parser.add_argument("--do_topics", action="store_true", help="Run topic modeling with BERTopic")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Sentence-Transformer for topics")
    parser.add_argument("--do_emotions", action="store_true", help="Run emotion recognition with emotion2vec+")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "json"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "txt"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "conllu"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "ner"), exist_ok=True)
    
    episode_path = args.episode_path
    episode_filename = os.path.basename(episode_path)
    base_name = os.path.splitext(episode_filename)[0]
    
    # Check if file has already been processed (skip if JSON exists)
    json_path = os.path.join(args.out_dir, f"{base_name}_whisperx.json")
    if os.path.exists(json_path):
        print(f"\n{'='*80}")
        print(f"Skipping: {episode_filename} (already processed)")
        print(f"  JSON file exists: {json_path}")
        print(f"{'='*80}\n")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing: {episode_filename}")
    print(f"{'='*80}\n")
    
    # Step 1: ASR with WhisperX
    print("Step 1: Running ASR (WhisperX)...")
    result, timings = transcribe_and_diarize(
        video_path=episode_path,
        hf_token=args.hf_token,
        output_dir=args.out_dir,
        asr_backend=args.asr_backend,
        asr_model_name=args.asr_model,
        diarization_model="pyannote/speaker-diarization-3.1",
        num_speakers=args.num_speakers,
        gpu_index=args.gpu_index,
        forced_language=args.language,  # Pass language to ensure alignment runs
    )
    
    # JSON is already saved by transcribe_and_diarize
    # json_path is already defined above (for skip check)
    print(f" JSON: {json_path}")
    
    # Step 2: Convert to TXT
    print("\nStep 2: Converting to TXT format...")
    txt_path = os.path.join(args.out_dir, "txt", f"{base_name}.txt")
    whisperx2txt(result, txt_path)
    
    # Step 3: Convert to CoNLL-U
    print("\nStep 3: Converting to CoNLL-U format...")
    conllu_path = os.path.join(args.out_dir, "conllu", f"{base_name}.conllu")
    whisperx2conllu(result, conllu_path, base_name, episode_filename, args.language)
    
    # Step 4: Run NER on CoNLL-U (only if CoNLL-U file was created)
    ner_metrics = None
    if not args.skip_ner:
        if os.path.exists(conllu_path):
            print("\nStep 4: Running NER on CoNLL-U...")
            ner_path = os.path.join(args.out_dir, "ner", f"{base_name}.entities.tsv")
            ner_metrics = run_ner_on_conllu(conllu_path, ner_path, ner_model=args.ner_model, gpu_index=args.gpu_index)
        else:
            print("\nStep 4: Skipping NER (CoNLL-U file not created - no segments with words found)")
    else:
        print("\nStep 4: Skipping NER (--skip_ner specified)")

    # Step 5: Emotion Recognition (optional)
    emotion_metrics = None
    if args.do_emotions:
        print("\nStep 5: Running Emotion Recognition (emotion2vec+)...")
        # Find WAV file (should be in wav/ subdirectory or same directory)
        wav_path = os.path.join(args.out_dir, "wav", f"{base_name}.wav")
        if not os.path.exists(wav_path):
            # Try same directory
            wav_path = os.path.join(args.out_dir, f"{base_name}.wav")
        if not os.path.exists(wav_path):
            # Try original episode path if it's a WAV
            if episode_path.lower().endswith('.wav'):
                wav_path = episode_path
        
        emotion_metrics = run_emotion_recognition(
            result.get("segments", []),
            wav_path,
            json_path,
            gpu_index=args.gpu_index
        )
    
    # Step 6: Topic Modeling (optional)
    topics_metrics = None
    if args.do_topics:
        print("\nStep 6: Running Topic Modeling (BERTopic)...")
        topics_dir = os.path.join(args.out_dir, "topics")
        topics_metrics = run_topic_modeling_from_segments(
            result.get("segments", []), 
            topics_dir, 
            base_name,
            embedding_model_name=args.embedding_model,
            gpu_index=args.gpu_index
        )
    
    print(f"\n{'='*80}")
    print("Pipeline complete!")
    print(f"{'='*80}")
    print(f"Outputs:")
    print(f"  JSON:   {json_path}")
    print(f"  TXT:    {txt_path}")
    print(f"  CoNLL-U: {conllu_path}")
    if not args.skip_ner:
        print(f"  NER:    {os.path.join(args.out_dir, 'ner', f'{base_name}.entities.tsv')}")
        if isinstance(ner_metrics, dict):
            print(f"  NER time: {ner_metrics.get('ner_elapsed_s', 'n/a')} s")
            alloc_b = ner_metrics.get('ner_max_mem_alloc_bytes')
            reserv_b = ner_metrics.get('ner_max_mem_reserved_bytes')
            if alloc_b is not None:
                print(f"  NER peak GPU allocated: {alloc_b/(1024**3):.3f} GB")
            #if reserv_b is not None:
            #    print(f"  NER peak GPU reserved: {reserv_b/(1024**3):.3f} GB")
    if args.do_emotions:
        print(f"  Emotions: Added to JSON segments")
        if emotion_metrics and isinstance(emotion_metrics, dict):
            print(f"  Emotion recognition time: {emotion_metrics.get('emotion_elapsed_s', 'n/a')} s")
            print(f"  Emotion distribution: {emotion_metrics.get('emotion_counts', {})}")
    if args.do_topics:
        print(f"  Topics: {os.path.join(args.out_dir, 'topics')} (CSV/HTML)")
        if topics_metrics and isinstance(topics_metrics, dict):
            print(f"  Topic modeling time: {topics_metrics.get('topics_elapsed_s', 'n/a')} s")
            alloc_b = topics_metrics.get('topics_max_mem_alloc_bytes')
            reserv_b = topics_metrics.get('topics_max_mem_reserved_bytes')
            if alloc_b is not None:
                print(f"  Topic modeling peak GPU allocated: {alloc_b/(1024**3):.3f} GB")
            if reserv_b is not None:
                print(f"  Topic modeling peak GPU reserved: {reserv_b/(1024**3):.3f} GB")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

