#!/bin/bash

BWA="./bwa-0.7.17/bwa"

VERBOSE=false
while getopts r:q:o:v FLAG
do
  case "${FLAG}" in
    r) REF=${OPTARG};;  # Path to reference genome
    q) FASTQ=${OPTARG};;  # Path to FASTQ file
    o) OUTPATH=${OPTARG};;  # Output file (full path or relative path)
    v) VERBOSE=true;;
    *) echo "$(basename $0): Invalid command line option: -$FLAG" ;;
  esac
done

FNAME=$(basename $FASTQ)
EXT="${FASTQ##*.}"
FNAME=$(basename $FASTQ .$EXT)

TMP="$FNAME"_filt.fq
# Select reads that contain only A,C,G,T (N's are a bug in ART Illumina read generation)
cat $FASTQ | awk '{
        if (NR%4==1) {L1=$0} 
        else if(NR%4==2) { if($0 ~ "^[ACGT][ACGT]+[ACGT]$") {L2=$0} else {L2="invalid"} }  
        else if(NR%4==3) {L3=$0} 
        else { if(L2!="invalid") {print L1"\n"L2"\n"L3"\n"$0} }
        }' > $TMP

# $BWA index $REF
$BWA mem -t 5 $REF $TMP | samtools view -F 4 > $OUTPATH 2> /dev/null  # Align and remove unmapped reads

OUTNAME=$(basename $OUTPATH ".sam")
awk '{print $1" "$4}' $OUTPATH > $OUTNAME"_pos.txt"  # Save read names and positions