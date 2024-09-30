#!/bin/bash
# Concatenate all the viral genomes in the iCAV folder
# Processing appends the folder name to the header to enable tracking of the source of the genome

# Assumed directory structure
# -concat_genomes.fa (current script)
# -iCAV_all_data
#   --virus1
#     ---virus1_genomes.fasta
#   --virus2
#     ---virus2_genomes.fasta

ICAV="iCAV_all_data"  # Path to unzipped iCAV data folder
OUT="viral_genomes.fa"  # Output file

for VDIR in "$ICAV"/* ; do 
  if [ -d $VDIR ]; then
    echo $VDIR
    VIR=$(basename $VDIR)
    for FASTA in "$VDIR"/*".fasta" ; do
      if [ -f $FASTA ]; then
        echo $FASTA
        cat $FASTA | awk -v var="$VIR" '{if ($0 ~ /^>/) {print ">"var"_"substr($0,2) } else {print $0}}' > $OUT
      fi
    done
  fi
done