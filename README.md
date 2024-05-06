# Acknowledgement

This repo is  for the 6000E course project. Thanks professor Kim and TAs for your guidance. The data procession part is inspired by  [LMOps/adaptllm at main · microsoft/LMOps (github.com)](https://github.com/microsoft/LMOps/tree/main/adaptllm), and the fine-tuning & evaluation part are mainly based on the LLaMa Factory [hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs (github.com)](https://github.com/hiyouga/LLaMA-Factory). Thanks for their work.

# 1. Install dependencies
You can change to other sources if the installing process is slow.

```bash
git clone https://github.com/mjy2357/6000E_biomedical_chatbot.git
conda create -n 6000E_biomedical_chatbot python==3.10
conda activate 6000E_biomedical_chatbot
cd 6000E_biomedical_chatbot
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
sudo apt-get update
sudo apt-get install unrar
```

# 2. Prepare data

Firstly, download and unpackage 'data' folder (including filtered biomedical subset of pile: '03_med.json'). If these commands fail, you may need download the 'data.rar' from https://huggingface.co/datasets/dango2357/6000E_project manually and **place it under the directory**.

```bash
wget https://huggingface.co/datasets/dango2357/6000E_project/resolve/main/data.rar
unrar x data.rar
```

Then, sample 100,000 instances and convert the format of data from raw to reading comprehension (This may take around 10 minutes)

```bash
python raw2read.py
```

# 3. Set training configurations and fine-tune the model

Firstly, download and unpackage the base model 'Llama-2-7b-chat-hf' folder. If these commands fail, you may need download the 'Llama-2-7b-chat-hf.rar' from https://huggingface.co/datasets/dango2357/6000E_project manually and **place it under the directory**.

```bash
wget https://huggingface.co/datasets/dango2357/6000E_project/resolve/main/Llama-2-7b-chat-hf.rar
unrar x Llama-2-7b-chat-hf.rar
```

**Use following commands on the terminal to fine-tune the base model. (The training process maybe quite long, we strongly suggest using the trained LoRA adapter directly).**

This code block is for fine-tuning on the pile_med dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path Llama-2-7b-chat-hf \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir data \
    --dataset pile_med \
    --cutoff_len 5120 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 50 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --output_dir saves/LLaMA2-7B-Chat/lora/train_on_pilemed_1epoch \
    --fp16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --plot_loss True
```

This code block is for fine-tuning on the ai-medical-chatbot dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path Llama-2-7b-chat-hf \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir data \
    --dataset ai-medical-chatbot_train \
    --cutoff_len 5120 \
    --learning_rate 1e-04 \
    --num_train_epochs 1.0 \
    --max_samples 250000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 50 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --output_dir saves/LLaMA2-7B-Chat/lora/train_on_medicalchatbot_1epoch \
    --fp16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --plot_loss True
```



# 4. Evaluation

**To convert the format of the output and facilitate the evaluation, we sampled a small number of instances from corresponding trainingsets to further fine-tune all the base model and fine-tuned models (MedQA 100 samples for 5 epochs and PubMedQA 100 samples for 1 epoch).**

This code block is for prediction on the **MedQA** testset of model **fine-tuned on the pile_med** dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path Llama-2-7b-chat-hf \
    --adapter_name_or_path saves/LLaMA2-7B-Chat/lora/train_on_medqa_ft_5epoch_pilemed_100000 \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir data \
    --dataset medqa_test \
    --cutoff_len 5120 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/LLaMA2-7B-Chat/lora/eval_on_medqa_sft_on_pile_med \
    --do_predict True

```

This code block is for prediction on the **PubMedQA** testset of model **fine-tuned on the pile_med dataset:**

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path Llama-2-7b-chat-hf \
    --adapter_name_or_path saves/LLaMA2-7B-Chat/lora/train_on_pubmedqa_ft_1epoch_pilemed_100000 \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir data \
    --dataset pubmedqa_test \
    --cutoff_len 5120 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/LLaMA2-7B-Chat/lora/eval_on_pubmedqa_sft_on_pile_med \
    --do_predict True
```

This code block is for prediction on the **MedQA** testset of model **fine-tuned on the ai-medical-chatbot dataset:**

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path Llama-2-7b-chat-hf \
    --adapter_name_or_path saves/LLaMA2-7B-Chat/lora/train_on_medqa_ft_5epoch_aimedicalchatbot_all \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir data \
    --dataset medqa_test \
    --cutoff_len 5120 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/LLaMA2-7B-Chat/lora/eval_on_medqa_sft_on_medicalchatbot \
    --do_predict True
```

This code block is for prediction on the **PubMedQA** testset of model **fine-tuned on the ai-medical-chatbot dataset:**

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path Llama-2-7b-chat-hf \
    --adapter_name_or_path saves/LLaMA2-7B-Chat/lora/train_on_pubmedqa_ft_1epoch_aimedicalchatbot_all \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir data \
    --dataset pubmedqa_test \
    --cutoff_len 5120 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/LLaMA2-7B-Chat/lora/eval_on_pubmedqa_sft_on_medicalchatbot \
    --do_predict True
```

After getting the generated results, you can use this command to compute accuracy.

```bash
python evaluation.py
```

