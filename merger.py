#!/usr/bin/env python3
import pandas as pd
import glob
import os
import argparse

def merge_csvs(input_folder, output_file):
    # Find all CSV files in the given folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in '{input_folder}'")
        return

    # Read each CSV into a DataFrame
    df_list = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            df_list.append(df)
        except Exception as e:
            print(f"Skipping '{fp}': read error: {e}")

    if not df_list:
        print("No valid CSV files to merge.")
        return

    # Concatenate all DataFrames, aligning columns by name
    combined = pd.concat(df_list, ignore_index=True)

    # Save the merged CSV
    combined.to_csv(output_file, index=False)
    print(f"Merged {len(df_list)} files from '{input_folder}' into '{output_file}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge multiple CSV files (with identical columns) into one.')
    parser.add_argument('input_folder',
                        help='Path to folder containing CSV files to merge')
    parser.add_argument('output_file',
                        help='Path for the merged output CSV file')
    args = parser.parse_args()
    merge_csvs(args.input_folder, args.output_file)
