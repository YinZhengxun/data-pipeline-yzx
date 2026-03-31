#!/usr/bin/env python3
"""
Topic detection using Qwen model for each audio file.
Extracts all sentences from a JSON file, groups by audio_filepath,
and assigns a topic category to each sentence.
"""

import os
import json
import re
import time
import sys
import argparse
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allowed topic categories (must match the prompt exactly)
ALLOWED_TOPICS = [
    "Music", "Sports", "Science", "Travel", "News", "Films", 
    "Gaming", "Comedy", "Vehicles", "Animals", "other"
]

# Topic prompt template
TOPIC_PROMPT = """
You are a video topic classification expert.

Analyze the following video transcription and determine which single category best describes its main topic.

Possible categories (choose only one):
Sports, Science, Travel, News, Films, Gaming, Comedy, Vehicles, Animals, Music, Other

Guidelines:
- Carefully read the full text.
- Select the category that BEST fits the main subject or theme.
- If none clearly fit, return "Other".
- Do NOT guess based on order or choose the first option by default.
- Return ONLY the category name, exactly as written above. No punctuation, no explanation, no JSON.
"""

def normalize_topic(topic: str) -> Optional[str]:
    """
    Normalize topic string to match allowed topics.
    Returns the matched topic or None if no match.
    """
    topic = topic.strip()
    
    # Handle common typos/variants
    if topic.lower() in ["aminals", "animal"]:
        return "Animals"
    if topic.lower() in ["film", "movie", "movies"]:
        return "Films"
    
    # Direct match (case-insensitive)
    for allowed in ALLOWED_TOPICS:
        if topic.lower() == allowed.lower():
            return allowed
    
    # Try to find partial match
    topic_lower = topic.lower()
    for allowed in ALLOWED_TOPICS:
        if allowed.lower() in topic_lower or topic_lower in allowed.lower():
            return allowed
    
    return None


def extract_topic_from_response(response: str) -> Optional[str]:
    """
    Extract topic from model response.
    Tries to find one of the allowed topics in the response.
    """
    response = response.strip()
    
    # Try direct normalization
    normalized = normalize_topic(response)
    if normalized:
        return normalized
    
    # Try to find topic in the response (case-insensitive search)
    response_lower = response.lower()
    for topic in ALLOWED_TOPICS:
        if topic.lower() in response_lower:
            # Make sure it's a word boundary match
            pattern = r'\b' + re.escape(topic.lower()) + r'\b'
            if re.search(pattern, response_lower):
                return topic
    
    return None


def run_topic_detection(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    transcript_text: str,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
    verbose: bool = False,
) -> str:
    """
    Run topic detection using Qwen model.
    """
    # Combine prompt and transcript
    # Use a more structured format to reduce first-item bias
    full_prompt = f"""{TOPIC_PROMPT}

Video Transcription:
{transcript_text}

Category:"""
    
    if verbose:
        print(f"    [DEBUG] Full prompt being sent to Qwen:")
        print(f"    {'='*60}")
        print(f"    {full_prompt}")
        print(f"    {'='*60}")
    
    messages = [
        {"role": "user", "content": full_prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
        )
    
    output_ids = generated[0][inputs.input_ids.shape[1]:]
    out_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return out_text.strip()


def detect_topic_with_retry(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    transcript_text: str,
    max_retries: int = 2,
    verbose: bool = False,
) -> str:
    """
    Detect topic with retry logic.
    Returns topic name or "other2" if validation fails after retries.
    """
    for attempt in range(max_retries):
        try:
            response = run_topic_detection(model, tokenizer, transcript_text, verbose=verbose and attempt == 0)
            if verbose:
                print(f"    [DEBUG] Qwen raw response: {response}")
            topic = extract_topic_from_response(response)
            
            if topic:
                return topic
            
            # If first attempt failed, try again
            if attempt < max_retries - 1:
                if verbose:
                    print(f"    [DEBUG] Invalid response, retrying... (attempt {attempt + 2}/{max_retries})")
                continue
                
        except Exception as e:
            print(f"  Error in topic detection (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                continue
    
    # All attempts failed or returned invalid topic
    return "other2"


def group_segments_by_file(segments: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group segments by audio_filepath if available, otherwise treat all as one file.
    Returns dict mapping file identifier to list of segments.
    """
    grouped = defaultdict(list)
    has_audio_filepath = any(seg.get("audio_filepath") for seg in segments)
    
    if has_audio_filepath:
        # Group by audio_filepath (for speaker JSON files)
        for segment in segments:
            audio_filepath = segment.get("audio_filepath", "")
            if audio_filepath:
                grouped[audio_filepath].append(segment)
    else:
        # All segments belong to the same file (whisperx JSON)
        grouped["all"] = segments
    
    return dict(grouped)


def extract_transcript_text(segments: List[Dict]) -> str:
    """
    Extract and combine text from segments into a single paragraph.
    """
    texts = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if text:
            texts.append(text)
    return " ".join(texts)


def process_json_file(
    json_path: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: Optional[str] = None,
) -> bool:
    """
    Process a JSON file: detect topics for each unique audio_filepath and add topic field.
    """
    print(f"Processing: {os.path.basename(json_path)}")
    
    # Read JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ERROR: Failed to read JSON file: {e}")
        return False
    
    # Handle both list of segments and dict with segments key
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
    else:
        print(f"  ERROR: Invalid JSON structure")
        return False
    
    if not segments:
        print(f"  WARNING: No segments found in file")
        return False
    
    # Group segments by audio_filepath
    grouped = group_segments_by_file(segments)
    print(f"  Found {len(grouped)} unique audio file(s)")
    
    # Process each file
    file_topics = {}
    for idx, (file_key, file_segments) in enumerate(grouped.items(), 1):
        if file_key == "all":
            print(f"  [{idx}/{len(grouped)}] Processing: all segments (whisperx format)")
        else:
            filename = os.path.basename(file_key)
            print(f"  [{idx}/{len(grouped)}] Processing: {filename}")
        
        # Extract transcript text
        transcript = extract_transcript_text(file_segments)
        if not transcript:
            print(f"    WARNING: No text found, using 'other2'")
            file_topics[file_key] = "other2"
            continue
        
        # Print transcript being sent to Qwen (truncated if too long)
        transcript_preview = transcript[:500] if len(transcript) > 500 else transcript
        print(f"    Transcript (first 500 chars): {transcript_preview}")
        if len(transcript) > 500:
            print(f"    ... (total length: {len(transcript)} characters)")
        print(f"    Full transcript length: {len(transcript)} characters, {len(transcript.split())} words")
        
        # Detect topic
        # Set verbose=True to see full prompt and response
        # Check both environment variable and command line argument
        verbose_env = os.getenv("QWEN_VERBOSE", "false").lower() == "true"
        # Note: verbose flag is passed via command line, environment variable is set in main()
        verbose = verbose_env
        topic = detect_topic_with_retry(model, tokenizer, transcript, verbose=verbose)
        file_topics[file_key] = topic
        print(f"    Topic: {topic}")
    
    # Add topic field to all segments
    if "all" in file_topics:
        # For whisperx JSON, all segments get the same topic
        topic = file_topics["all"]
        for segment in segments:
            segment["topic"] = topic
    else:
        # For speaker JSON, use audio_filepath as key
        for segment in segments:
            audio_filepath = segment.get("audio_filepath", "")
            if audio_filepath in file_topics:
                segment["topic"] = file_topics[audio_filepath]
            else:
                segment["topic"] = "other2"
    
    # Save updated JSON
    output_file = output_path or json_path
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f" Saved updated JSON to: {os.path.basename(output_file)}")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to save JSON: {e}")
        return False


def load_model(model_id: str, device_map: str = "auto", gpu_index: int = None):
    """Load Qwen model and tokenizer."""
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Set device map based on GPU index if provided
    if gpu_index is not None and isinstance(device_map, str) and device_map == "auto":
        device_map = f"cuda:{gpu_index}"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    print("Model loaded successfully")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Topic detection for JSON files using Qwen model")
    parser.add_argument("--json_file", required=True, help="Path to JSON file to process")
    parser.add_argument("--model_id", default="Qwen/Qwen3-0.6B", help="Qwen model ID")
    parser.add_argument("--output_file", default=None, help="Output JSON file path (default: overwrite input)")
    parser.add_argument("--device_map", default="auto", help="Device map for model loading")
    parser.add_argument("--gpu_index", type=int, default=None, help="GPU index (overrides device_map if provided)")
    parser.add_argument("--batch_dir", default=None, help="Process all JSON files in directory")
    parser.add_argument("--verbose", action="store_true", help="Print full prompt and response for debugging")
    
    args = parser.parse_args()
    
    # Set environment variable for verbose mode
    if args.verbose:
        os.environ["QWEN_VERBOSE"] = "true"
    
    # Load model once
    model, tokenizer = load_model(args.model_id, args.device_map, args.gpu_index)
    
    # Process single file or batch
    if args.batch_dir:
        # Process all JSON files in directory
        json_files = [f for f in os.listdir(args.batch_dir) if f.endswith('.json')]
        json_files.sort()
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            json_path = os.path.join(args.batch_dir, json_file)
            process_json_file(json_path, model, tokenizer)
            print()
    else:
        # Process single file
        if not os.path.exists(args.json_file):
            print(f"ERROR: File not found: {args.json_file}")
            sys.exit(1)
        
        try:
            success = process_json_file(args.json_file, model, tokenizer, args.output_file)
            if not success:
                print(f"ERROR: Failed to process {args.json_file}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Exception while processing {args.json_file}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

