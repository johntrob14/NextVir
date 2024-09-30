#!/bin/bash

# Script to count the lengths of genomes for each class (human/virus).

TMP="tmp"

# Statistics for human class
: <<'HUMAN'
grep -v '^>' references/GRCh38.fa > $TMP
NUMN=$(awk -v RS='N' 'END{print NR-1}' $TMP)  # Number of N's
echo "Number of N's in human genome: $NUMN"
NUMC=$(wc $TMP | awk '{print $3-$1}')  # Number of characters
echo "Length of human genome: $NUMC"
echo "True length of human genome: $((NUMC-NUMN))"

READS=$(grep '^@NC' human_reads_filt.fq | wc -l)  # Number of reads
echo "Number of human reads: $READS"
HUMAN

# Statistics for viral class
CLASS="MCV"
echo "Statistics for $CLASS"
GENOMES=$(grep "^>"$CLASS references/viral_genomes.fa | wc -l)
echo "Number of genomes: $GENOMES"

cat references/viral_genomes.fa | awk -v var="$CLASS" '{
        if (NR%2==1) { if($0 ~ "^>"var) {L1="valid"} else {L1="invalid"} }
        else { if(L1!="invalid") { print $0 } }
        }' > $TMP

NUMC=$(wc $TMP | awk '{print $3-$1}')  # Number of characters
echo "Total length of viral genomes: $NUMC"

READS=$(grep "^@"$CLASS viral_reads_filt.fq | wc -l)  # Number of reads
echo "Number of viral reads: $READS"

rm $TMP
