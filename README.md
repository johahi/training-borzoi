# Training Borzoi

This repository contains two scripts to either train Borzoi from scratch, or to fine-tune a pre-trained Borzoi or Flashzoi model on your own data or the Borzoi training data. Both scripts rely on huggingface accelerate for DDP training.
Note that you will have to provide your own dataloader (+eval function) for the Borzoi target files, as the webdataset approach I used is probably inferior to other implementations (such as adapting [these](https://github.com/yangzhao1230/Enformer_Borzoi_Training_Pytorch/tree/main/dataloaders) for the Borzoi training data).

## Borzoi from scratch

To train a Borzoi from scratch, you can adapt and run borzoi_from_scratch.py across 2 GPUs using hf-accelerate. I suggest using fp16 mixed-precision training, as I found bf16 not to converge as well.
The modeling directory used in the script was modified from a very early borzoi-pytorch version, but the weight dicts are directly compatible with the latest borzoi-pytorch. The version in this repo contains, as far as I know, all the initializations used in Keras where Borzoi was originally trained. Weight decay and optimizer settings are also set to be identical to the Keras ones in the script.

## Fine-tuning from Borzoi/Flashzoi

To fine-tune a Borzoi/Flashzoi model, you can adapt and run finetuning.py across 2 GPUs using hf-accelerate. This script should work with borzoi-pytorch.

## References

[Borzoi-pytorch](https://github.com/johahi/borzoi-pytorch/)  
[Borzoi implementation and weights](https://github.com/calico/borzoi).  
<a id="1">[1]</a>
Linder, Johannes, et al. "Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation." Nature Genetics (2025): 1-13; doi: [https://doi.org/10.1101/2023.08.30.555582](https://doi.org/10.1038/s41588-024-02053-6)  
<a id="2">[2]</a>
Flashzoi: An enhanced Borzoi model for accelerated genomic analysis  
Johannes C. Hingerl, Alexander Karollus, Julien Gagneur  
bioRxiv 2024.12.18.629121; doi: [https://doi.org/10.1101/2024.12.18.629121](https://www.biorxiv.org/content/10.1101/2024.12.18.629121v1)  

## Citation

Please cite the Borzoi paper [1], along with Flashzoi [2], if you used this repository or the models.
