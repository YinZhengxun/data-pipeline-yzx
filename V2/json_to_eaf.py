#!/usr/bin/env python3
"""
Convert DAVA pipeline outputs (JSON, NER, Topics) to ELAN EAF format.
Scans output directory for per-file folders and creates EAF files.
"""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

def milliseconds_to_elan_time(seconds: float) -> int:
    """Convert seconds to ELAN time format (milliseconds)."""
    return int(seconds * 1000)

def create_eaf_structure(media_file: Optional[str] = None) -> ET.Element:
    """Create the basic EAF XML structure."""
    # Create root element
    root = ET.Element("ANNOTATION_DOCUMENT")
    root.set("AUTHOR", "DAVA Pipeline")
    root.set("DATE", "")
    root.set("FORMAT", "3.0")
    root.set("VERSION", "3.0")
    
    # Header
    header = ET.SubElement(root, "HEADER")
    header.set("MEDIA_FILE", "")
    header.set("TIME_UNITS", "milliseconds")
    
    # Media descriptor
    if media_file:
        media_descriptor = ET.SubElement(header, "MEDIA_DESCRIPTOR")
        media_descriptor.set("MEDIA_URL", f"file://{os.path.abspath(media_file)}")
        media_descriptor.set("MIME_TYPE", "video/mp4")  # Will be adjusted based on file type
    
    # Time order (will be populated)
    time_order = ET.SubElement(root, "TIME_ORDER")
    
    # Linguistic types
    linguistic_types = ET.SubElement(root, "LINGUISTIC_TYPE_GRAPH")
    
    # Add default linguistic types
    for lt_id, lt_attrs in [
        ("default-lt", {"TIME_ALIGNABLE": "true", "GRAPHIC_REFERENCES": "false"}),
        ("transcription-lt", {"TIME_ALIGNABLE": "true", "GRAPHIC_REFERENCES": "false"}),
        ("word-lt", {"TIME_ALIGNABLE": "true", "GRAPHIC_REFERENCES": "false"}),
        ("speaker-lt", {"TIME_ALIGNABLE": "true", "GRAPHIC_REFERENCES": "false"}),
        ("ner-lt", {"TIME_ALIGNABLE": "true", "GRAPHIC_REFERENCES": "false"}),
        ("topic-lt", {"TIME_ALIGNABLE": "true", "GRAPHIC_REFERENCES": "false"}),
        ("emotion-lt", {"TIME_ALIGNABLE": "true", "GRAPHIC_REFERENCES": "false"}),
    ]:
        lt = ET.SubElement(linguistic_types, "LINGUISTIC_TYPE")
        lt.set("LINGUISTIC_TYPE_ID", lt_id)
        for key, value in lt_attrs.items():
            lt.set(key, value)
    
    # Locales
    locales = ET.SubElement(root, "LOCALE")
    locales.set("LANGUAGE_CODE", "en")
    locales.set("COUNTRY_CODE", "US")
    
    return root

def add_time_slot(time_order: ET.Element, time_value: int) -> str:
    """Add a time slot and return its ID."""
    time_slot_id = f"ts{time_value}"
    # Check if already exists
    for existing in time_order.findall("TIME_SLOT"):
        if existing.get("TIME_SLOT_ID") == time_slot_id:
            return time_slot_id
    
    time_slot = ET.SubElement(time_order, "TIME_SLOT")
    time_slot.set("TIME_SLOT_ID", time_slot_id)
    time_slot.set("TIME_VALUE", str(time_value))
    return time_slot_id

def create_tier(root: ET.Element, tier_id: str, linguistic_type: str, participant: str = "", 
                annotator: str = "", default_locale: str = "") -> ET.Element:
    """Create a tier element and return its annotations container."""
    # Check if tier already exists
    for existing_tier in root.findall("TIER"):
        if existing_tier.get("TIER_ID") == tier_id:
            annotations = existing_tier.find("ANNOTATION")
            if annotations is not None:
                return annotations
    
    # Create new tier
    tier = ET.SubElement(root, "TIER")
    tier.set("LINGUISTIC_TYPE_REF", linguistic_type)
    tier.set("TIER_ID", tier_id)
    if participant:
        tier.set("PARTICIPANT", participant)
    if annotator:
        tier.set("ANNOTATOR", annotator)
    if default_locale:
        tier.set("DEFAULT_LOCALE", default_locale)
    
    annotations = ET.SubElement(tier, "ANNOTATION")
    return annotations

def add_annotation(annotations_elem: ET.Element, annotation_id: str, 
                   start_time: int, end_time: int, text: str, 
                   time_order: ET.Element) -> ET.Element:
    """Add an alignable annotation."""
    alignable = ET.SubElement(annotations_elem, "ALIGNABLE_ANNOTATION")
    alignable.set("ANNOTATION_ID", annotation_id)
    
    # Get or create time slots
    start_slot_id = add_time_slot(time_order, start_time)
    end_slot_id = add_time_slot(time_order, end_time)
    
    alignable.set("TIME_SLOT_REF1", start_slot_id)
    alignable.set("TIME_SLOT_REF2", end_slot_id)
    
    annotation_value = ET.SubElement(alignable, "ANNOTATION_VALUE")
    annotation_value.text = text
    
    return alignable

def load_ner_entities(ner_path: str, segments: List[Dict]) -> List[Dict]:
    """Load NER entities from TSV file and map to time using segments."""
    entities = []
    if not os.path.exists(ner_path):
        return entities
    
    try:
        with open(ner_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            header = None
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                
                # Check if first line is header
                if line_idx == 0 and 'sent_id' in line.lower():
                    header = [p.lower() for p in parts]
                    continue
                
                # Parse based on header or default format
                if header:
                    # Has header - map columns
                    entity_dict = {}
                    for i, col_name in enumerate(header):
                        if i < len(parts):
                            entity_dict[col_name] = parts[i]
                    
                    # Map to time using sent_id and character positions
                    sent_id = entity_dict.get('sent_id', '')
                    entity_text = entity_dict.get('entity', '')
                    label = entity_dict.get('label', '')
                    
                    # Try to find corresponding segment and map character positions to time
                    try:
                        sent_idx = int(sent_id) - 1  # Convert to 0-based
                        if 0 <= sent_idx < len(segments):
                            seg = segments[sent_idx]
                            seg_text = seg.get('text', '')
                            char_start = int(entity_dict.get('start', 0))
                            char_end = int(entity_dict.get('end', len(seg_text)))
                            
                            # Map character positions to time (simplified - proportional)
                            seg_start = seg.get('start', 0.0)
                            seg_end = seg.get('end', seg_start + 1.0)
                            seg_duration = seg_end - seg_start
                            
                            if len(seg_text) > 0:
                                # Proportional mapping
                                start_ratio = char_start / len(seg_text)
                                end_ratio = char_end / len(seg_text)
                                time_start = seg_start + (start_ratio * seg_duration)
                                time_end = seg_start + (end_ratio * seg_duration)
                            else:
                                time_start = seg_start
                                time_end = seg_end
                            
                            entities.append({
                                'text': entity_text,
                                'type': label,
                                'start': time_start,
                                'end': time_end,
                            })
                    except (ValueError, IndexError):
                        # Skip if we can't map
                        continue
                else:
                    # No header - assume format: entity, type, start, end
                    if len(parts) >= 2:
                        entities.append({
                            'text': parts[0],
                            'type': parts[1],
                            'start': float(parts[2]) if len(parts) > 2 and parts[2].replace('.', '').isdigit() else 0.0,
                            'end': float(parts[3]) if len(parts) > 3 and parts[3].replace('.', '').isdigit() else 0.0,
                        })
    except Exception as e:
        print(f"  WARNING: Could not load NER file: {e}")
        import traceback
        traceback.print_exc()
    
    return entities

def load_topics(topics_csv_path: str) -> Dict[str, str]:
    """Load topics from CSV file."""
    topics_map = {}
    if not os.path.exists(topics_csv_path):
        return topics_map
    
    try:
        import csv
        import ast
        with open(topics_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle different CSV formats
                topic_name = row.get('Name', row.get('Topic', ''))
                if not topic_name:
                    continue
                
                # Try to get representative documents
                # Could be 'Representative_Docs' (plural) or 'Representative_Document' (singular)
                docs_str = row.get('Representative_Docs', row.get('Representative_Document', ''))
                
                if docs_str:
                    try:
                        # Try to parse as Python list string representation
                        docs_list = ast.literal_eval(docs_str)
                        if isinstance(docs_list, list):
                            # Map each document to the topic name
                            for doc in docs_list:
                                if doc and isinstance(doc, str):
                                    topics_map[doc] = topic_name
                        elif isinstance(docs_list, str):
                            topics_map[docs_list] = topic_name
                    except (ValueError, SyntaxError):
                        # If parsing fails, treat as single string
                        if docs_str:
                            topics_map[docs_str] = topic_name
    except Exception as e:
        print(f"  WARNING: Could not load topics file: {e}")
        import traceback
        traceback.print_exc()
    
    return topics_map

def convert_json_to_eaf(json_path: str, output_eaf_path: str, 
                        media_file: Optional[str] = None,
                        ner_path: Optional[str] = None,
                        topics_csv_path: Optional[str] = None,
                        base_name: str = ""):
    """Convert WhisperX JSON to EAF format."""
    
    print(f"Converting: {os.path.basename(json_path)}")
    
    # Load JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ERROR: Could not read JSON file: {e}")
        return False
    
    segments = data.get("segments", [])
    if not segments:
        print(f"  WARNING: No segments found in JSON")
        return False
    
    # Load additional data
    ner_entities = []
    if ner_path:
        ner_entities = load_ner_entities(ner_path, segments)
        print(f"  Loaded {len(ner_entities)} NER entities")
    
    topics_map = {}
    if topics_csv_path:
        topics_map = load_topics(topics_csv_path)
        print(f"  Loaded {len(topics_map)} topics")
    
    # Check for emotions in segments
    emotion_count = sum(1 for seg in segments if seg.get("emotion"))
    if emotion_count > 0:
        print(f"  Found {emotion_count} segments with emotion data")
    
    # Create EAF structure
    root = create_eaf_structure(media_file)
    time_order = root.find("TIME_ORDER")
    
    # Get all unique speakers
    speakers = set()
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        if speaker:
            speakers.add(speaker)
        # Also check words
        for word in seg.get("words", []):
            word_speaker = word.get("speaker")
            if word_speaker:
                speakers.add(word_speaker)
    
    speakers = sorted(list(speakers))
    print(f"  Found {len(speakers)} speakers: {', '.join(speakers)}")
    
    # Create tiers
    # 1. Main transcription tier (all segments)
    transcription_ann = create_tier(root, "Transcription", "transcription-lt")
    
    # 2. Word-level tier
    words_ann = create_tier(root, "Words", "word-lt")
    
    # 3. Speaker tiers (one per speaker)
    speaker_tiers = {}
    for speaker in speakers:
        speaker_ann = create_tier(root, f"Speaker_{speaker}", "speaker-lt", participant=speaker)
        speaker_tiers[speaker] = speaker_ann
    
    # 4. NER tier (if available)
    ner_ann = None
    if ner_entities:
        ner_ann = create_tier(root, "Named_Entities", "ner-lt")
    
    # 5. Topics tier (if available)
    topics_ann = None
    if topics_map:
        topics_ann = create_tier(root, "Topics", "topic-lt")
    
    # 6. Emotion tier (if available in segments)
    emotion_ann = None
    has_emotions = any(seg.get("emotion") for seg in segments)
    if has_emotions:
        emotion_ann = create_tier(root, "Emotions", "emotion-lt")
    
    # Process segments
    annotation_counter = 0
    
    for seg_idx, segment in enumerate(segments):
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        text = segment.get("text", "").strip()
        speaker = segment.get("speaker", "UNKNOWN")
        
        if not text:
            continue
        
        start_ms = milliseconds_to_elan_time(start_time)
        end_ms = milliseconds_to_elan_time(end_time)
        
        # Add segment to transcription tier
        ann_id = f"a{annotation_counter}"
        add_annotation(transcription_ann, ann_id, start_ms, end_ms, text, time_order)
        annotation_counter += 1
        
        # Add to speaker tier
        if speaker in speaker_tiers:
            ann_id = f"a{annotation_counter}"
            add_annotation(speaker_tiers[speaker], ann_id, start_ms, end_ms, text, time_order)
            annotation_counter += 1
        
        # Add words
        words = segment.get("words", [])
        for word_data in words:
            word_text = word_data.get("word", "").strip()
            word_start = word_data.get("start", start_time)
            word_end = word_data.get("end", end_time)
            
            if not word_text:
                continue
            
            word_start_ms = milliseconds_to_elan_time(word_start)
            word_end_ms = milliseconds_to_elan_time(word_end)
            
            ann_id = f"a{annotation_counter}"
            add_annotation(words_ann, ann_id, word_start_ms, word_end_ms, word_text, time_order)
            annotation_counter += 1
        
        # Add topics if available (simplified - map by time)
        if topics_ann is not None and topics_map and len(topics_map) > 0:
            # Try to find topic for this segment (simplified matching)
            segment_text_lower = text.lower()
            matched = False
            for topic_text, topic_name in topics_map.items():
                # More flexible matching: check if key words from topic text appear in segment
                topic_words = [w.strip() for w in topic_text.lower().split() if len(w.strip()) > 3]
                if topic_words and any(word in segment_text_lower for word in topic_words[:3]):
                    try:
                        ann_id = f"a{annotation_counter}"
                        add_annotation(topics_ann, ann_id, start_ms, end_ms, topic_name, time_order)
                        annotation_counter += 1
                        matched = True
                        break
                    except Exception as e:
                        print(f"  WARNING: Failed to add topic annotation: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Add emotion if available
        if emotion_ann is not None:
            emotion = segment.get("emotion")
            emotion_id = segment.get("emotion_id")
            emotion_score = segment.get("emotion_score")
            
            if emotion:
                # Format: "emotion_name (score: 0.XX)" or just "emotion_name"
                if emotion_score is not None:
                    emotion_text = f"{emotion} (score: {emotion_score:.3f})"
                else:
                    emotion_text = emotion
                
                try:
                    ann_id = f"a{annotation_counter}"
                    add_annotation(emotion_ann, ann_id, start_ms, end_ms, emotion_text, time_order)
                    annotation_counter += 1
                except Exception as e:
                    print(f"  WARNING: Failed to add emotion annotation: {e}")
    
    # Add NER entities
    if ner_ann is not None and ner_entities and len(ner_entities) > 0:
        for entity in ner_entities:
            try:
                entity_start = milliseconds_to_elan_time(entity.get('start', 0.0))
                entity_end = milliseconds_to_elan_time(entity.get('end', entity.get('start', 0.0) + 1.0))
                entity_text = f"{entity.get('text', '')} [{entity.get('type', 'UNKNOWN')}]"
                
                ann_id = f"a{annotation_counter}"
                add_annotation(ner_ann, ann_id, entity_start, entity_end, entity_text, time_order)
                annotation_counter += 1
            except Exception as e:
                print(f"  WARNING: Failed to add NER entity '{entity.get('text', '')}': {e}")
    
    # Write EAF file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    
    try:
        tree.write(output_eaf_path, encoding='utf-8', xml_declaration=True)
        print(f"  ✓ EAF file created: {os.path.basename(output_eaf_path)}")
        print(f"    Total annotations: {annotation_counter}")
        return True
    except Exception as e:
        print(f"  ERROR: Could not write EAF file: {e}")
        return False

def find_file_in_dir(directory: str, pattern: str) -> Optional[str]:
    """Find a file matching pattern in directory."""
    dir_path = Path(directory)
    for ext in ['', '.json', '.tsv', '.csv']:
        file_path = dir_path / f"{pattern}{ext}"
        if file_path.exists():
            return str(file_path)
    
    # Also try without extension in filename
    for file_path in dir_path.glob(f"*{pattern}*"):
        if file_path.is_file():
            return str(file_path)
    
    return None

def process_output_directory(out_dir: str, output_eaf_dir: Optional[str] = None):
    """Process all per-file folders in output directory.
    
    Expected structure:
    out_dir/
      folder1/
        folder1_whisperx.json
      folder2/
        folder2_whisperx.json
    """
    
    out_path = Path(out_dir)
    if not out_path.exists():
        print(f"ERROR: Output directory does not exist: {out_dir}")
        return
    
    print("="*60)
    print("DAVA to EAF Converter")
    print("="*60)
    print(f"Scanning: {out_dir}")
    print()
    
    # Find all subdirectories (each should contain a JSON file named {folderName}_whisperx.json)
    subdirs = [d for d in out_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print("No subdirectories found!")
        return
    
    print(f"Found {len(subdirs)} subdirectories to process")
    print()
    
    successful = 0
    failed = 0
    skipped = 0
    
    for folder_path in sorted(subdirs):
        folder_name = folder_path.name
        
        # Look for JSON file: {folderName}_whisperx.json
        json_file = folder_path / f"{folder_name}_whisperx.json"
        
        if not json_file.exists():
            print(f"\n[{successful + failed + skipped + 1}/{len(subdirs)}] Skipping: {folder_name}")
            print(f"  JSON file not found: {json_file.name}")
            skipped += 1
            continue
        
        # Output EAF path (save in same folder)
        output_eaf_path = folder_path / f"{folder_name}.eaf"
        
        # Look for associated files in the same folder
        media_file = None
        for ext in ['.mp4', '.mkv', '.webm', '.m4a', '.mp3', '.wav']:
            media_candidate = folder_path / f"{folder_name}{ext}"
            if media_candidate.exists():
                media_file = str(media_candidate)
                break
        
        ner_path = find_file_in_dir(str(folder_path), f"{folder_name}.entities")
        topics_csv_path = find_file_in_dir(str(folder_path), f"{folder_name}_topics")
        
        print(f"\n[{successful + failed + skipped + 1}/{len(subdirs)}] Processing: {folder_name}")
        if media_file:
            print(f"  Media file: {os.path.basename(media_file)}")
        if ner_path:
            print(f"  NER file: {os.path.basename(ner_path)}")
        if topics_csv_path:
            print(f"  Topics file: {os.path.basename(topics_csv_path)}")
        
        if convert_json_to_eaf(
            str(json_file),
            str(output_eaf_path),
            media_file=media_file,
            ner_path=ner_path,
            topics_csv_path=topics_csv_path,
            base_name=folder_name
        ):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(subdirs)}")
    print(f"\nEAF files saved in their respective folders")

def main():
    parser = argparse.ArgumentParser(
        description="Convert DAVA pipeline outputs to ELAN EAF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all files in output directory
  python json_to_eaf.py --out_dir /path/to/output

  # Convert single JSON file
  python json_to_eaf.py --json_file file.json --output file.eaf

  # Convert with media file reference
  python json_to_eaf.py --json_file file.json --output file.eaf --media_file video.mp4
        """
    )
    
    parser.add_argument("--out_dir", type=str, help="Output directory with per-file folders")
    parser.add_argument("--json_file", type=str, help="Single JSON file to convert")
    parser.add_argument("--output", "-o", type=str, help="Output EAF file path")
    parser.add_argument("--media_file", type=str, help="Media file path (for single file mode)")
    parser.add_argument("--ner_file", type=str, help="NER entities TSV file (for single file mode)")
    parser.add_argument("--topics_file", type=str, help="Topics CSV file (for single file mode)")
    parser.add_argument("--eaf_output_dir", type=str, help="Directory for EAF files (default: out_dir/eaf)")
    
    args = parser.parse_args()
    
    if args.json_file:
        # Single file mode
        if not args.output:
            base_name = Path(args.json_file).stem.replace("_whisperx", "")
            args.output = f"{base_name}.eaf"
        
        convert_json_to_eaf(
            args.json_file,
            args.output,
            media_file=args.media_file,
            ner_path=args.ner_file,
            topics_csv_path=args.topics_file
        )
    elif args.out_dir:
        # Directory mode
        process_output_directory(args.out_dir, args.eaf_output_dir)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

