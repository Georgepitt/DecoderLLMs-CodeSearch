# Zero-shot

In the Zero-shot testing section, we provide all the necessary scripts for testing with CSN and CoSQA$^+$. The Python scripts prefixed with `CSN_Test` are for CSN testing, while those prefixed with `CoSQA_Plus` are for CoSQA$^+$. Below, we demonstrate how to use these scripts.

## CSN Test
For the CSN tests, we provide examples for three different models: Decoder_Model, UniXcoder, and CodeBERT. The usage is similar for each model. Here’s an example for the Decoder_Model:

```
cd decoder-only-code-search/Zero-shot

python CSN_Test_Decoder_Model.py \
    --model_name_or_path google/codegemma-7b-it \
    --result_path CSN-codegemma \
    --test_data_path_dir ../Dataset/CodeSearchNet \
    --embedding_batch_size 500
```

To use different models, simply change the script and the model path accordingly. If you want to reproduce the results from our paper "Are Decoder-Only Large Language Models the Silver Bullet for Code Search?", you can find the download links for the models referenced in Table II of the paper in Section III, STUDY SETUP.

Before running the scripts, you need to download the datasets from [this Google Drive link](). Particularly, the CSN dataset has been processed to remove low-quality data.

Here’s an example for running the UniXcoder script:

```
python CSN_Test_UniXcoder.py \
    --model_name_or_path microsoft/unixcoder-base \
    --result_path CSN-UniXcoder \
    --test_data_path_dir ../Dataset/CodeSearchNet \
    --embedding_batch_size 500
```

And for CodeBERT:

```
python CSN_Test_CodeBERT.py \
    --model_name_or_path microsoft/codebert-base \
    --result_path CSN-codebert \
    --test_data_path_dir ../Dataset/CodeSearchNet \
    --embedding_batch_size 500
```

The example output format includes results for six languages: Python, Go, JavaScript, Java, PHP, and Ruby. Here’s a sample output:

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

## CoSQA_Plus Test
For CoSQA$^+$, the testing procedure is similar. Modify the script and model path accordingly. Here’s an example for the Decoder_Model:

```
python CoSQA_Plus_Test_Decoder_Model.py \
    --model_name_or_path google/codegemma-7b-it \
    --result_path CoSQA_plus-codegemma \
    --test_data_path_dir ../Dataset/CoSQA_Plus \
    --embedding_batch_size 500
```

Example output:

```
Loading checkpoint shards: 100%|██████████| 3/3 [00:05<00:00,  1.78s/it]
07/15/2024 21:39:17 - INFO - __main__ -   loading data...
Shape of data_code: (51516,)
Each batch includes 500 data points
100%|██████████| 20604/20604 [00:30<00:00, 674.08it/s]
07/15/2024 21:42:34 - INFO - __main__ -   ***** Eval results *****
07/15/2024 21:42:34 - INFO - __main__ -     eval_Map = 0.0
07/15/2024 21:42:34 - INFO - __main__ -     eval_mrr = 0.0
07/15/2024 21:42:34 - INFO - __main__ -     time = 0.0
eval_Map: 0.0004004354810678825
eval_mrr: 0.00040890722862771335
```

For CodeBERT, you can use:

```
python CoSQA_Plus_Test_CodeBERT.py \
    --model_name_or_path microsoft/codebert-base \
    --result_path CoSQA_Plus-codebert \
    --test_data_path_dir ../Dataset/CoSQA_Plus \
    --embedding_batch_size 500
```

And for UniXcoder:

```
python CoSQA_Plus_Test_UniXcoder.py \
    --model_name_or_path microsoft/unixcoder-base \
    --result_path CoSQA_Plus-UniXcoder \
    --test_data_path_dir ../Dataset/CoSQA_Plus \
    --embedding_batch_size 500
```






















