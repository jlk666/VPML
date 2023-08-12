#!/bin/bash
count=$(find /home/zhuosl/VPML/Genome_annotation -type f -name "*.gff" | wc -l)
echo "The total number of prokka annotation gff is $count"
