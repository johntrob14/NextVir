#!/bin/bash

ART="art_bin_MountRainier/art_illumina"

VERBOSE=false
while getopts f:o:n:v FLAG
do
  case "${FLAG}" in
    f) REF=${OPTARG};;  # Path to reference genome
    o) OUTPATH=${OPTARG};;  # Output file (full path or relative path)
    n) READCNT=${OPTARG};;  # Number of reads to be generated
    v) VERBOSE=true;;
    *) echo "$(basename $0): Invalid command line option: -$FLAG" ;;
  esac
done

if ! [[ -d $(dirname $OUTPATH) ]];then
  mkdir $(dirname $OUTPATH)
fi


NUM_GENOME=$(grep -o '>' $REF | wc -l)
RCNT=`expr $READCNT / $NUM_GENOME`
# echo $READCNT
# echo $NUM_GENOME
echo $RCNT
# $ART -na -q -i $REF -l 150 --seqSys HS25 --rcount $RCNT -o $OUTPATH
$ART -na -q -i $REF -l 150 --seqSys MSv3 --rcount $RCNT -o $OUTPATH -nf 1
