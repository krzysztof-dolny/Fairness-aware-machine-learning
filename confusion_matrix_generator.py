import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import math
import os

# Parse arguments: size of dataset (n) and length of confusion matrix vector (k)
parser = argparse.ArgumentParser(description="Generate confusion matrix combinations.")
parser.add_argument("n", type=int, help="First numerical parameter - dataset size (n)")
parser.add_argument("k", type=int, nargs="?", default=8,
                    help="Second numerical parameter - confusion matrix length (k) (default is 8)")
args = parser.parse_args()

# Validate input
n = args.n
k = args.k

if n <= 0 or k <= 0:
    raise ValueError("n and k must be > 0 and <= 255")

# Output path setup
cm_file_path = "outputs/confusion_matrix.parquet"
os.makedirs(os.path.dirname(cm_file_path), exist_ok=True)

# Global state for ParquetWriter
pqwriter = None
first_write = True


def save_data_to_parquet(result, file_path):
    """Save confusion matrix combinations to a Parquet file."""
    global pqwriter, first_write
    df = pd.DataFrame(result)
    table = pa.Table.from_pandas(df)
    
    if first_write:
        pqwriter = pq.ParquetWriter(file_path, table.schema)
        first_write = False
    
    pqwriter.write_table(table)


def generate_confusion_matrix(n, k, file_path):
    """Generate all possible confusion matrix combinations."""
    def recursive_generate(remaining, depth, current_combination):
        nonlocal result
        
        if depth == k:
            if remaining == 0:
                result.append(np.array(current_combination, dtype=np.int8))

                # Save data in chunks of 10 million rows monitor progress
                if len(result) == 10_000_000:
                    save_data_to_parquet(result, file_path)
                    result = []
                    print("Saved next 10M records")
            return
        
        max_value = remaining
        for value in range(max_value + 1):
            recursive_generate(remaining - value, depth + 1, current_combination + [value])
    
    result = []
    recursive_generate(n, 0, [])
    
    if result:
        save_data_to_parquet(result, file_path)


def calculate_num_of_cm_combinations(n, k):
    """Calculating number of confusion matrix combinations from formula."""
    C = [n + k - 1, k - 1]
    combinations = math.factorial(C[0]) / (math.factorial(C[1]) * math.factorial(C[0] - C[1]))
    return int(combinations)


if __name__ == "__main__":
    # Calculating number of possible confusion matrix combinations
    number_of_combinations = calculate_num_of_cm_combinations(n, k)
    print(f"Number of possible confusion matrix combinations = {number_of_combinations}")

    # Generate all possible confusion matrix combinations and save to Parquet file
    generate_confusion_matrix(n, k, cm_file_path)

    # Close pqwriter
    if pqwriter:
        pqwriter.close()
