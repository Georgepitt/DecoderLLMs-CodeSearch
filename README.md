# Are Decoder-Only Large Language Models the Silver Bullet for Code Search?

This repository contains the code and datasets for the paper "Are Decoder-Only Large Language Models the Silver Bullet for Code Search?" Our work is divided into three main parts: zero-shot tests with decoder-only LLMs, fine-tuning tests with decoder-only LLMs, and improvement analysis. This repository provides the necessary code and data to reproduce our results.

Each section has its own dedicated directory containing all relevant scripts. Below, we provide an overview and demonstration example for each section.

## Experimental Environment

**Hardware:**

- **CPU:** Intel(R) Xeon(R) Platinum 8360H CPU @ 3.00GHz
- **GPU:** 2 × NVIDIA A800 80GB GPUs
- **RAM:** 2.0 TB

**Software:**

- **Operating System:** CentOS Linux release 7.9.2009 (Core)
- **Python:** 3.8.19
- **PyTorch Version:** 2.3.0+cu121
- **CUDA Version:** 12.1

## Dependencies

To install the necessary dependencies, run the following commands:

```
cd decoder-only-code-search
pip install -r requirements.txt
```

## Datasets

The datasets can be accessed via [this Google Drive link](https://drive.google.com/drive/folders/1yhw_WKw72Fn4izkPIAmy5s_LdiwDc9hg?usp=sharing). The dataset structure is as follows:

```
Dataset
   |__CodeSearchNet
   |__CoSQA_Plus
   |__Train
        |__CSN
        |__E5
        |__MNTP
        |__SimCSE
```

## Zero-Shot Test

All scripts for zero-shot code search are located in the [Zero-shot](https://github.com/Georgepitt/decoder-only-code-search/tree/main/Zero-shot) directory. These scripts measure distances using cosine similarity. Below is an example of testing CodeGemma on the CodeSearchNet dataset. Additional examples can be found in the same directory.

```
cd decoder-only-code-search/Zero-shot

python CSN_Test_Decoder_Model.py \
    --model_name_or_path google/codegemma-7b-it \
    --result_path CSN-codegemma \
    --test_data_path_dir ../Dataset/CodeSearchNet \
    --embedding_batch_size 500
```

Example output:

```
Loading checkpoint shards: 100%|██████████| 3/3 [00:05<00:00,  1.78s/it]
Evaluating language: python
Shape of data_code: (22176,)
Each batch contains 500 data 
Processing batches: 100%|██████████| 45/45 [04:01<00:00,  5.38s/it]
python MRR Score: 0.10966641818108162
Evaluating language: go
......
```

## Fine-Tuning Test

All scripts for fine-tuning code search models are in the [Fine-tuning](https://github.com/Georgepitt/decoder-only-code-search/tree/main/Fine-tuning) directory. These scripts also use cosine similarity to measure distances. Below is an example of fine-tuning CodeGemma on the CodeSearchNet dataset. More examples can be found in the Fine-tuning directory. Note that before running the fine-tuning test, the model needs to be fine-tuned. Detailed instructions can be found in the [Fine-tuning Method](https://github.com/Georgepitt/decoder-only-code-search/tree/main/Fine-tuning/Fine-tuning_Method) directory.

```
cd decoder-only-code-search/Fine-tuning

python CSN_Test_Finetuning_Decoder_Model.py \
    --model_name_or_path google/codegemma-7b-it \
    --peft_model_name_or_path finetuning_model \
    --result_path CSN-finetuning-codegemma \
    --test_data_path_dir ../Dataset/CodeSearchNet \
    --embedding_batch_size 500
```

## Improvement Analysis

All scripts for improvement analysis are provided in the [Improvement Analysis](https://github.com/Georgepitt/decoder-only-code-search/tree/main/Improvement_Analysis) directory.
