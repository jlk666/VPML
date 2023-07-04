#!/bin/bash
folder="/home/zhuosl/VPML/Genome/genome"  # Replace with the actual folder path
csv_file="/home/zhuosl/VPML/Genome/genome_list.csv"  # Replace with the actual CSV file path

# Get the file names with "fna.gz" extension in the folder
ls -R "$folder" | grep -E "fna\.gz$" > "$csv_file"

# Modify the file names in the CSV to keep only the information before "_genomic.fna.gz"
sed -i 's/_genomic\.fna\.gz$//' "$csv_file"

# Count the number of samples in the modified CSV
sample_count=$(wc -l < "$csv_file")

# Print the sample count
echo "Number of samples in $csv_file: $sample_count"
