rule download_genome:
    message: 
        "Downloading Vibrio parahaemolyticus COMPLETE genome from Refseq"
    
    shell:
    """
        python Genome_catcher.py
    """
    conda:
        "~/miniconda3/envs/VPML.yml"  