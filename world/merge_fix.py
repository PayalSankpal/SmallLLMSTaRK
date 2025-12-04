import pandas as pd
import glob
import os
import sys

# --- CONFIGURATION ---
# Point this to the folder containing your process_*.csv files
# Example: OUTPUT_DIR = "./output/PRIME_1_SCORE"
if len(sys.argv) > 1:
    OUTPUT_DIR = sys.argv[1]
else:
    OUTPUT_DIR = "."  # Current directory

def force_merge():
    print(f"Scanning directory: {OUTPUT_DIR}")
    
    # 1. Merge Full Data Dumps
    dump_files = glob.glob(os.path.join(OUTPUT_DIR, "full_dump_process_*.csv"))
    print(f"Found {len(dump_files)} dump files.")
    
    if dump_files:
        df_list = []
        for f in dump_files:
            try:
                # Read each file
                d = pd.read_csv(f)
                df_list.append(d)
            except Exception as e:
                print(f"Skipping bad file {f}: {e}")
        
        if df_list:
            full_df = pd.concat(df_list, ignore_index=True)
            # Remove duplicates if any (based on ID)
            full_df.drop_duplicates(subset=['id'], inplace=True)
            
            output_path = os.path.join(OUTPUT_DIR, "repaired_full_data_dump.csv")
            full_df.to_csv(output_path, index=False)
            print(f"SUCCESS: Repaired dump saved to {output_path}")
            print(f"Total Queries in Dump: {len(full_df)}")
        else:
            print("No valid data found in dump files.")

    # 2. Merge Pipeline Results
    res_files = glob.glob(os.path.join(OUTPUT_DIR, "pipeline_results_process_*.csv"))
    print(f"\nFound {len(res_files)} pipeline result files.")
    
    if res_files:
        res_list = []
        for f in res_files:
            try:
                d = pd.read_csv(f)
                res_list.append(d)
            except Exception as e:
                print(f"Skipping bad file {f}: {e}")
                
        if res_list:
            full_res = pd.concat(res_list, ignore_index=True)
            full_res.drop_duplicates(subset=['query_id'], inplace=True)
            
            output_path = os.path.join(OUTPUT_DIR, "repaired_pipeline_results.csv")
            full_res.to_csv(output_path, index=False)
            print(f"SUCCESS: Repaired results saved to {output_path}")
            print(f"Total Scored Queries: {len(full_res)}")

if __name__ == "__main__":
    force_merge()