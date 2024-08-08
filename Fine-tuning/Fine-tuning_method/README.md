# Fine-tuning_method

This directory includes training scripts for UniXcoder and CodeBERT. Additionally, it contains three fine-tuning methods—MNTP, SimCSE, and SupCon—specifically for decoder-only models. Due to the extensive content related to these methods, they are organized into separate folders: `MNTP`, `SimCSE`, and `SupCon`. Each folder contains the relevant training scripts and instructions. Below, we demonstrate the training processes for UniXcoder and CodeBERT.

## UniXcoder

To train UniXcoder, use the following command:

```
cd decoder-only-code-search/Fine-tuning/Fine-tuning_method

python UniXcoder_run.py \
    --output_dir saved_models/UniXcoder \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --train_data_file ../../Dataset/Train/SimCSE/CSN/Encoder_train/Combine_train.jsonl \
    --eval_data_file ../../Dataset/Train/SimCSE/CSN/Encoder_train/python_codebase.jsonl \
    --codebase_file ../../Dataset/Train/SimCSE/CSN/Encoder_train/python_codebase.jsonl \
    --num_train_epochs 2 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456  
```

### CodeBERT

To train CodeBERT, use the following command:

```
cd decoder-only-code-search/Fine-tuning/Fine-tuning_method

python CodeBERT_run.py \
    --output_dir saved_models/CodeBERT \
    --model_name_or_path microsoft/codebert-base  \
    --do_train \
    --train_data_file ../../Dataset/Train/SimCSE/CSN/Encoder_train/Combine_train.jsonl \
    --eval_data_file ../../Dataset/Train/SimCSE/CSN/Encoder_train/python_codebase.jsonl \
    --codebase_file ../../Dataset/Train/SimCSE/CSN/Encoder_train/python_codebase.jsonl \
    --num_train_epochs 2 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456  
```



