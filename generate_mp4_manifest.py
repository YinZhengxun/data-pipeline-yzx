#!/usr/bin/env python3
"""
Generate CSV manifest file with file paths of a specific type.
Searches a directory recursively and writes one file path per line.

Supports: mp4, mp3, wav, flac, and other extensions.
"""

import argparse
import os
from pathlib import Path

# Supported file extensions
SUPPORTED_EXTENSIONS = ["mp4", "mp3", "wav", "flac", "m4a", "aac", "ogg", "flv", "avi", "mkv"]

def find_files(root_dir, extension):
    """
    Recursively find all files with the given extension in the directory.
    
    Args:
        root_dir: Directory to search
        extension: File extension (without dot, e.g., "mp4", "wav")
    
    Returns:
        List of absolute file paths
    """
    files = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"ERROR: Directory does not exist: {root_dir}")
        return []
    
    extension_lower = extension.lower()
    extension_upper = extension.upper()
    
    print(f"Searching for {extension_upper} files in: {root_dir}")
    
    # Find all files with the extension (case-insensitive search)
    for file_path in root_path.rglob(f"*.{extension_lower}"):
        files.append(str(file_path.resolve()))
    
    for file_path in root_path.rglob(f"*.{extension_upper}"):
        files.append(str(file_path.resolve()))
    
    # Also try capitalized version (e.g., "Mp4")
    if extension_lower != extension_upper:
        extension_capitalized = extension_lower.capitalize()
        for file_path in root_path.rglob(f"*.{extension_capitalized}"):
            files.append(str(file_path.resolve()))
    
    # Remove duplicates (in case of case-insensitive filesystem)
    files = sorted(list(set(files)))
    
    return files

def write_manifest(files, output_file):
    """Write file paths to CSV file (one per line)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in files:
            f.write(f"{file_path}\n")
    
    print(f"✓ Manifest written: {output_file}")
    print(f"  Total files: {len(files)}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV manifest file with file paths of a specific type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Generate MP4 manifest
  {__file__} --dir /path/to/videos --ext mp4

  # Generate WAV manifest
  {__file__} --dir /path/to/audio --ext wav

  # Generate MP3 manifest with custom output
  {__file__} --dir /path/to/audio --ext mp3 --output my_mp3_manifest.csv

Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}
        """
    )
    
    parser.add_argument(
        "--dir", "--directory",
        type=str,
        required=True,
        help="Directory to search recursively for files"
    )
    
    parser.add_argument(
        "--ext", "--extension",
        type=str,
        required=True,
        help=f"File extension to search for (without dot, e.g., 'mp4', 'wav'). Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV file path (default: <extension>_manifest.csv in current directory)"
    )
    
    args = parser.parse_args()
    
    # Normalize extension (remove dot if present, convert to lowercase)
    extension = args.ext.lstrip('.').lower()
    
    # Validate extension
    if extension not in SUPPORTED_EXTENSIONS:
        print(f"WARNING: Extension '{extension}' is not in the supported list.")
        print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        response = input(f"Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = f"{extension}_manifest.csv"
    
    print("=" * 60)
    print("File Manifest Generator")
    print("=" * 60)
    print(f"Search directory: {args.dir}")
    print(f"File extension: {extension}")
    print(f"Output file: {output_file}")
    print()
    
    # Find all files
    files = find_files(args.dir, extension)
    
    if not files:
        print("WARNING: No files found!")
        return
    
    # Write to CSV
    write_manifest(files, output_file)
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"First few files:")
    for i, file_path in enumerate(files[:5], 1):
        print(f"  {i}. {file_path}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

if __name__ == "__main__":
    main()

