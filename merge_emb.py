import os
import re
import torch
from pathlib import Path

def merge_batch_embeddings(
    output_path: str = "emb/mag/text-embedding-ada-002/doc/candidate_emb_dict_new.pt",
    batch_pattern: str = "mag_openai_emb_*.pt",
    batch_dir: str = "/home/payal-s/BtechProject/stark"
):
    """
    Merge batch embedding files into the main candidate_emb_dict.pt file.
    
    Args:
        output_path: Path to the main embedding dictionary file
        batch_pattern: Pattern to match batch files (glob pattern)
        batch_dir: Directory containing batch files
    """
    
    # Load existing embeddings or create new dict
    output_file = Path(output_path)
    if output_file.exists():
        print(f"Loading existing embeddings from {output_path}")
        candidate_emb_dict = torch.load(output_path)
        print(f"Existing entries: {len(candidate_emb_dict)}")
    else:
        print(f"Creating new embedding dictionary")
        candidate_emb_dict = {}
        # Create directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all batch files
    batch_dir_path = Path(batch_dir)
    batch_files = sorted(batch_dir_path.glob(batch_pattern))
    
    if not batch_files:
        print(f"No batch files found matching pattern: {batch_dir} / {batch_pattern}")
        return
    
    print(f"\nFound {len(batch_files)} batch files:")
    for bf in batch_files:
        print(f"  - {bf.name}")
    
    # Extract batch numbers and sort
    batch_file_info = []
    for bf in batch_files:
        match = re.search(r'_(\d+)\.pt$', bf.name)
        if match:
            batch_num = int(match.group(1))
            batch_file_info.append((batch_num, bf))
    
    batch_file_info.sort(key=lambda x: x[0])
    
    # Merge batches
    total_new_entries = 0
    total_updated_entries = 0
    
    print("\nMerging batches...", batch_file_info)
    for batch_num, batch_file in batch_file_info:
        print(f"\nProcessing batch {batch_num}: {batch_file.name}")
        
        try:
            batch_dict = torch.load(batch_file)
            
            if not isinstance(batch_dict, dict):
                print(f"  WARNING: {batch_file.name} is not a dictionary, skipping")
                continue
            
            print(f"  Entries in batch: {len(batch_dict)}")
            
            new_entries = 0
            updated_entries = 0
            
            for key, value in batch_dict.items():
                if key in candidate_emb_dict:
                    updated_entries += 1
                else:
                    new_entries += 1
                candidate_emb_dict[key] = value
            
            print(f"  New entries: {new_entries}")
            print(f"  Updated entries: {updated_entries}")
            
            total_new_entries += new_entries
            total_updated_entries += updated_entries
            
        except Exception as e:
            print(f"  ERROR loading {batch_file.name}: {e}")
            continue
    
    # Save merged dictionary
    print(f"\n{'='*60}")
    print(f"Merge Summary:")
    print(f"  Total new entries added: {total_new_entries}")
    print(f"  Total entries updated: {total_updated_entries}")
    print(f"  Final dictionary size: {len(candidate_emb_dict)}")
    print(f"\nSaving merged embeddings to {output_path}")
    
    torch.save(candidate_emb_dict, output_path)
    print(f"✓ Successfully saved merged embeddings")
    
    # Print sample keys
    if candidate_emb_dict:
        sample_keys = list(candidate_emb_dict.keys())[:5]
        print(f"\nSample keys in merged dictionary:")
        for key in sample_keys:
            print(f"  - {key}")
    
    return candidate_emb_dict


if __name__ == "__main__":
    # Run the merge
    merged_dict = merge_batch_embeddings()
    
    print(f"\n{'='*60}")
    print("Merge complete!")
