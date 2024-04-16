#!/bin/bash

cd ~

pwd

# Directory containing the files to process
input_dir=~/projects/kva/kva_data/kva_parsed_jsons_pymupdf_simple

# Python script to process each file
python_script="/home/sandra/git/KVA-terminiprojekt/db_processing/chunk_and_vectorize_json.py"

# Configutation file path
config_file="/home/sandra/git/KVA-terminiprojekt/db_processing/chunk_and_vectorize_config.json"

# Logging files path
log_dir="projects/kva/kva_data/logs/logs_kva_simple_data/"

# Iterate over all files in the input directory
for file in "$input_dir"/*; do
    echo "$input_dir"
    echo $file
    # Call the Python script with the current file as input and the output file as output
    python "$python_script" "$config_file" "$file" "$log_dir"
done
