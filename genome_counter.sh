#!/bin/bash
folder="/home/zhuosl/VPML/Genome/genome"  # Replace with the actual folder path

# Count the number of files in the folder
file_count=$(ls -l "$folder" | grep -v "^d" | wc -l)

# Print the file count
echo "Number of files in $folder: $file_count"