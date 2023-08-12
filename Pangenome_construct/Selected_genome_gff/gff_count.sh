#!/bin/bash
count=$(find /home/zhuosl/VPML/Pangenome_construct/Selected_genome_gff -type f -name "*.gff" | wc -l)
echo "The total number of selected prokka annotation gff is $count"
