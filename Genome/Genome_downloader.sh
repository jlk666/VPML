esearch -db assembly -query "Vibrio parahaemolyticus[Organism]" | efetch -format docsum | xtract -pattern DocumentSummary -element AssemblyAccession > accessions.txt

ncbi-acc-download -F fasta -o downloaded_genomes accessions.txt
