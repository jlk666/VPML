#!/bin/bash
folder="/home/zhuosl/VPML/Genome/genome"  # Replace with the actual folder path

# Count the number of files with the "fna.gz" extension in the folder
file_count=$(find "$folder" -type f -name "*.fna.gz" | wc -l)

# Print the file count
echo "Number of fna.gz files in $folder: $file_count"
