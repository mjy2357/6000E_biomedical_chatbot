---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /hpc2hdd/home/jmiao996/LLaMA-Factory/Llama-2-7b-chat-hf
model-index:
- name: train_on_pubmedqa_ft_1epoch_pilemed100000_1epoch
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_on_pubmedqa_ft_1epoch_pilemed100000_1epoch

This model is a fine-tuned version of [/hpc2hdd/home/jmiao996/LLaMA-Factory/Llama-2-7b-chat-hf](https://huggingface.co//hpc2hdd/home/jmiao996/LLaMA-Factory/Llama-2-7b-chat-hf) on the pubmedqa_ft dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 1.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.10.0
- Transformers 4.39.3
- Pytorch 2.2.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2