import pandas as pd
import argparse
 
def merge_parquet_files(file1, file2, output_file, merge_type="row"):
 
    print(f"Loading {file1}...")
    df1 = pd.read_parquet(file1)
 
    print(f"Loading {file2}...")
    df2 = pd.read_parquet(file2)
 
    merged_df = pd.concat([df1, df2], axis=0)
 
    print(f"Saving merged dataset to {output_file}...")
    import os
    output_path = os.path.dirname(output_file)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    merged_df.to_parquet(output_file, index=False)
    print("Merge complete!")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two Parquet files into one.")
    parser.add_argument("--file1", type=str, help="Path to the first Parquet file.")
    parser.add_argument("--file2", type=str, help="Path to the second Parquet file.")
    parser.add_argument("--output_file", type=str, help="Path to the output Parquet file.")
 
    args = parser.parse_args()
 
    merge_parquet_files(args.file1, args.file2, args.output_file)