# NextVir
As of 9/30, this repository has been made public for the submission of our manuscript: NextVir: Enabling Classification of Tumor-Causing Viruses with Genomic Foundation Models for publication.

## Requirements
Please use conda to install the main requirements. You may need to change the version of pytorch and/or HuggingFace depending on your hardware. All experiments were performed without triton, which may install automatically.

All experiments were performed using AMD Vega 20 GPUs. Support for mps and cpu backend is coming soon, but expect to need at least 8Gb of VRAM.

## Main
main.py is the entry point for training and testing models. In it's default configuration, main.py will train a NextVir-D model on multiclass icav data. The dataset will be available on zenodo shortly, but is a constructed set of 150bp reads and modified labels from 7 known oncoviruses and the human reference genome.

While documentation and reporting is still in progress, please see the full set of command-line arguments can be seen in utils/train_utils.py, in the parse_args() function.

Training was performed using DataParalel. To use a single gpu, pass `--device=0` to main, for multiple, use `--device=0,1,2,3`.

## Acknowledgments
The LoRA implementation is based on Hayden Prarie's CLIP finetuning work here: https://github.com/Hprairie/finetune-clip.
