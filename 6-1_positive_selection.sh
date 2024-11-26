#!/bin/bash

# find all available protein sequences in ./data/evo/00proseqs
# and align them using MUSCLE
# store the aligned sequences to ./data/evo/01proseqs_align

# link muscle command to local executable
muscle=/home/ceballos/muscle-linux-x86.v5.2

# create folder ./data/evo/01proseqs_align if not already created
mkdir -p ./data/evo/01proseqs_align

# find all protein sequences in ./data/evo/00proseqs and find what the gene name is
# find all .fasta files in ./data/evo/00proseqs
for file in ./data/evo/00proseqs/*.fasta
do
    fn=$(basename $file .fasta)
    printf "$fn\n"
    gene=$(echo $fn | cut -d'_' -f1)
    printf "Aligning $gene\n"
    $muscle -align ./data/evo/00proseqs/${gene}_proseq.fasta -output ./data/evo/01proseqs_align/${gene}_proseq_outv2.fasta
    printf "Done aligning $gene\n"
done

# reorder the sequences in ./data/evo/01proseqs_align and put them in ./data/evo/02proseqs_ordered
conda activate nnt
python 6-1-1_reorder_sequences.py

# create folder ./data/evo/03codalign if not already created
mkdir -p ./data/evo/03codalign


# activate conda environment with pal2nal
conda activate evo_test

# find all available protein sequences in 02proseqs and translate them to codons
for file in ./data/evo/02proseqs_ordered/*.fasta
do 
    fn=$(basename $file .fasta)
    printf "$fn\n"
    gene=$(echo $fn | cut -d'_' -f1)
    printf "Translating $gene\n"
    pal2nal.pl ./data/evo/02proseqs_ordered/${gene}_proseq_ordered.fasta ./data/evo/00nucseqs/${gene}_nucseq.fasta -output fasta > ./data/evo/03codalign/${gene}_codalign.fa
    printf "Done translating $gene\n"
done

# fill in missing sequences in ./data/evo/03codalign
conda activate nnt
python 6-1-2_fill_missing_sequences.py

# order the sequences in ./data/evo/04codalign_filled and save in ./data/evo/05codalign_taxnames
source_directory="./data/evo/04codalign_filled"
destination_directory="./data/evo/05codalign_taxnames"

# create the destination directory if it doesn't exist
mkdir -p "$destination_directory"

# define the patterns for the species you want to keep in the header
species_patterns="hsapiens|ptroglodytes|mmulatta|mmusculus|btaurus|dnovemcinctus|sharrisii|oanatinus|ggallus|xtropicalis|drerio|ccarcharias|pmarinus"

# loop through each oxt_codalign.fasta file in the source directory
for file in "$source_directory"/*.fa; do
    # get the filename without the directory path
    filename=$(basename "$file")

    # process the file and save it in the destination directory
    awk -v species="$species_patterns" '/^>/{ match($0, species); if (RSTART) print ">" substr($0, RSTART, RLENGTH); next }1' "$file" > "$destination_directory/$filename"

    echo "Processed $file and saved to $destination_directory/$filename"
done

# use Hyphy to model positive selection across a self-defined phylogenetic tree
conda activate evo_test

# create folder 06absrel_results if not already created
mkdir -p ./data/evo/06absrel_results

# find all .fa files in ./data/evo/05codalign_taxnames
for file in ./data/evo/05codalign_taxnames/*.fa
do
    fn=$(basename $file .fasta)
    printf "$fn\n"
    gene=$(echo $fn | cut -d'_' -f1)
    hyphy absrel --alignment ./data/evo/05codalign_taxnames/${gene}_codalign.fa --tree ./data/evo/timetree_bi.nwk --output ./data/evo/06absrel_results/${gene}_ABSREL.json
done
