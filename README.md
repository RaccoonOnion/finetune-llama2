# Instruction Fine-Tuning with LoRA

## Code

1. finetune-4bit.py: Finetuning
2. generate.py: Inference

First, install dependencies specified in requirements.txt

`conda create -n lora python=3.10`

`conda activate lora`

`pip install -r requirements.txt`

In order to run finetuning, just run
`python finetune-4bit.py`

For more options, run
`python finetune-4bit.py --base_model 'meta-llama/Llama-2-7b-hf' --data_path 'yahma/alpaca-cleaned' --output_dir './lora-alpaca' --batch_size 128 --micro_batch_size 4 --num_epochs 3 --lora_r 8 --resume_from_checkpoint './lora-alpaca/checkpoint-776'`

In order to run inference, just run
`python generate.py`

For more options, run
`python generate.py --base_model 'meta-llama/Llama-2-7b-hf' --lora_weights 'RaccoonOnion/Llama-2-7b-hf-LoRA-alpaca-cleaned-rank8' --prompt_template alpaca`

Note that rank8 weight is more recommended but you can also try rank1 weights. The checkpoint name for rank1 and rank8 weights are listed below.

## LoRA Weights
I uploaded two LoRA weights (one for rank 8, the other one for rank 1) to my Huggingface account.

Rank 8 checkpoint name: RaccoonOnion/Llama-2-7b-hf-LoRA-alpaca-cleaned-rank8
Rank 1 checkpoint name: RaccoonOnion/Llama-2-7b-hf-LoRA-alpaca-cleaned-rank1

By the time you review this document, they should be all public.

I already incorporate these two files into the generate.py settings. Hence there is no need to download the weights.

## Log File
I logged the loss every 10 update steps as required. There are three log files for each LoRA weights, each for one epoch. They are present in the HuggingFace repo **Files and Versions** as well as the log folder of this GitHub repo.

## About Experiment

The base model used is the official checkpoint for Llama2 on HuggingFace, checkpoint name: **meta-llama/Llama-2-7b-hf**.

The finetuning data used is the cleaned version of alpaca: **yahma/alpaca-cleaned**

Finetuning and inference are conducted on Nvidia V100 machine on PACE.


