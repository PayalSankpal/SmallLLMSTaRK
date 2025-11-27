import torch
import os
from pathlib import Path
from typing import List, Union
import argparse

def merge_pt_files(
    input_files: List[Union[str, Path]], 
    output_file: Union[str, Path],
    conflict_strategy: str = "first"
) -> dict:
    """
    Merge multiple .pt files into a single file.
    
    Args:
        input_files: List of paths to .pt files to merge
        output_file: Path where the merged .pt file will be saved
        conflict_strategy: How to handle conflicts ("first" or "last")
                         - "first": Keep the first occurrence (default)
                         - "last": Keep the last occurrence
    
    Returns:
        Dictionary with merge statistics
    """
    merged_dict = {}
    stats = {
        "total_files": len(input_files),
        "total_keys": 0,
        "conflicts": 0,
        "unique_keys": 0
    }
    
    print(f"Starting merge of {len(input_files)} files...")
    print(f"Conflict strategy: {conflict_strategy}\n")
    
    for i, file_path in enumerate(input_files, 1):
        print(f"Processing file {i}/{len(input_files)}: {file_path}")
        
        try:
            # Load the .pt file
            data = torch.load(file_path, map_location='cpu')
            
            # Handle different data structures
            if isinstance(data, dict):
                current_dict = data
            else:
                print(f"  Warning: File contains non-dict data (type: {type(data)})")
                print(f"  Wrapping in dict with key 'data_{i}'")
                current_dict = {f"data_{i}": data}
            
            # Merge dictionaries
            before_count = len(merged_dict)
            for key, value in current_dict.items():
                stats["total_keys"] += 1
                
                if key in merged_dict:
                    stats["conflicts"] += 1
                    if conflict_strategy == "last":
                        merged_dict[key] = value
                        print(f"  Conflict: Key '{key}' - keeping LAST occurrence")
                    else:
                        print(f"  Conflict: Key '{key}' - keeping FIRST occurrence")
                else:
                    merged_dict[key] = value
            
            after_count = len(merged_dict)
            new_keys = after_count - before_count
            print(f"  Added {new_keys} new keys (total now: {after_count})\n")
            
        except Exception as e:
            print(f"  Error loading {file_path}: {e}\n")
            continue
    
    stats["unique_keys"] = len(merged_dict)
    
    # Save merged dictionary
    print(f"Saving merged file to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(merged_dict, output_file)
    
    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Total keys found: {stats['total_keys']}")
    print(f"Unique keys in output: {stats['unique_keys']}")
    print(f"Conflicts encountered: {stats['conflicts']}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")
    
    return stats


def merge_pt_files_from_directory(
    directory: Union[str, Path],
    output_file: Union[str, Path],
    pattern: str = "*.pt",
    conflict_strategy: str = "first"
) -> dict:
    """
    Merge all .pt files from a directory.
    
    Args:
        directory: Directory containing .pt files
        output_file: Path where the merged .pt file will be saved
        pattern: Glob pattern for files to include (default: "*.pt")
        conflict_strategy: How to handle conflicts ("first" or "last")
    
    Returns:
        Dictionary with merge statistics
    """
    directory = Path(directory)
    pt_files = sorted(directory.glob(pattern))
    
    if not pt_files:
        print(f"No files matching '{pattern}' found in {directory}")
        return {}
    
    print(f"Found {len(pt_files)} files in {directory}\n")
    return merge_pt_files(pt_files, output_file, conflict_strategy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple .pt files into a single file.")
    parser.add_argument("input_files", nargs='+', help="List of .pt files to merge")
    parser.add_argument("-o", "--output_file", required=True, help="Path where the merged .pt file will be saved")
    parser.add_argument("-c", "--conflict_strategy", default="first", 
                        choices=["first", "last"], help="How to handle conflicts (default: first)")
    
    args = parser.parse_args()
    
    stats = merge_pt_files(args.input_files, args.output_file, args.conflict_strategy)
    print(stats)
