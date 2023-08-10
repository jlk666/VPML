#!/bin/bash

# Source and target directory
SRC_DIR="/home/zhuosl/VPML/Genome_annotation"
TARGET_DIR="/home/zhuosl/VPML/Pangenome_construct/Selected_genome_gff"

# Read the CSV file line by line
while IFS= read -r line
do
    # Construct the source path for each .fna file
    FILE_PATH="$SRC_DIR/prokka_$line/$line.gff"

    # Check if the file exists
    if [[ -f "$FILE_PATH" ]]; then
        # Move the file to the target directory
        cp "$FILE_PATH" "$TARGET_DIR"
    else
        # Print a message if the file does not exist
        echo "File $FILE_PATH does not exist"
    fi
done < genome_list_selected.csv
