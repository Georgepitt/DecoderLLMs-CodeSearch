import json
import torch
import os
import numpy as np
import time
import sys
import argparse

from transformers import AutoTokenizer, AutoModel, AutoConfig, RobertaTokenizer, RobertaModel, RobertaConfig
from peft import PeftModel

from typing import Optional, List, Dict, Any, NamedTuple, Iterable, Tuple
from more_itertools import chunked, flatten
from scipy.spatial.distance import cdist

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
class CodeBertModel(nn.Module):
    def __init__(self, encoder):
        super(CodeBertModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)



def read_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    if line.strip():  
                        data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {line.strip()}")
                    print(f"JSONDecodeError: {e}")
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
    return data



def compute_ranks(src_representations: np.ndarray,
                  tgt_representations: np.ndarray,
                  distance_metric: str) -> Tuple[np.array, np.array]:
    distances = cdist(src_representations, tgt_representations,
                      metric=distance_metric)
    # By construction the diagonal contains the correct elements
    correct_elements = np.expand_dims(np.diag(distances), axis=-1)
    return np.sum(distances <= correct_elements, axis=-1), distances



def compute_evaluation_metrics(code_representation,docstring_representation,test_batch_size,distance_metric):
    assert len(code_representation) > 0, 'data must have more than 0 rows.'
    assert len(docstring_representation) > 0, 'data must have more than 0 rows.'
    assert len(code_representation) == len(docstring_representation)
    np.random.seed(0)  # set random seed so that random things are reproducible
     # Shuffle the indices
    idxs = np.arange(len(code_representation))
    np.random.shuffle(idxs)
    # Apply the shuffled indices to both representations
    random_code_representation = code_representation[idxs]
    random_docstring_representation = docstring_representation[idxs]

    max_samples = 50
    sum_mrr = 0.0
    num_batches = 0
    
    batched_random_code_representation = list(chunked(list(random_code_representation), test_batch_size)) 
    batched_random_docstring_representation = list(chunked(list(random_docstring_representation), test_batch_size))

    for batch_idx, (batch_code, batch_doc) in enumerate(zip(batched_random_code_representation, batched_random_docstring_representation)):
        if len(batch_code) < test_batch_size:
            break  # the last batch is smaller than the others, exclude.
        num_batches += 1
        ranks, distances = compute_ranks(batch_code,
                                             batch_doc,
                                             distance_metric)
        sum_mrr += np.mean(1.0 / ranks)
        # print(f"Batch number: {num_batches}, Sum_MRR: {sum_mrr}")
    eval_mrr = sum_mrr / num_batches

    return eval_mrr

def load_representations(file_path: str) -> np.ndarray:
    return np.load(file_path)

def tokenize_and_convert_to_ids(text_list, max_length=200):

    all_token_ids = []

    for text in text_list:
        cleaned_text = ' '.join(text.split())
        tokens = tokenizer.tokenize(cleaned_text)[:max_length - 2] 
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]  
        token_ids = tokenizer.convert_tokens_to_ids(tokens)  
        padding_length = max_length - len(token_ids)
        token_ids += [tokenizer.pad_token_id] * padding_length  

        all_token_ids.append(token_ids)
    
    return torch.tensor(all_token_ids, dtype=torch.long)

def codebert_embedding(data,  character,model):
    inputs0 = tokenize_and_convert_to_ids(data)
    inputs = inputs0.to(device)  
    with torch.no_grad():
        if character == 'code_inputs':
            return model(code_inputs=inputs)
        elif character == 'nl_inputs':
            return model(nl_inputs=inputs)
        else:
            raise ValueError("character should be either 'code_inputs' or 'nl_inputs'")


def get_model_embedding(date_p,embedding_batch_size,character, model):
    # Batch data
    batch_chunked_data = chunked(date_p, embedding_batch_size) 
    print(f" Each batch contains {embedding_batch_size} data ")
    # Initializes a list to store the embedded results of all batches
    embeddings_list = []
    sum_batch_number = 0
    total_batches = len(date_p) // embedding_batch_size + (1 if len(date_p) % embedding_batch_size != 0 else 0)
    for batch_data in tqdm(batch_chunked_data, total=total_batches, desc="Processing batches"):
        batch_data_representation = codebert_embedding(batch_data,character,model)
        embeddings_list.append(batch_data_representation.cpu().numpy())
        sum_batch_number = sum_batch_number +1
        # logging.debug(f"Embedding batch {sum_batch_number}, total {total_batches} batches")
    # Stack the embedded results of all batches into a matrix
    matrix_presentation = np.vstack(embeddings_list)
    
    return matrix_presentation

def save_representations(data: np.ndarray, file_path: str):
    np.save(file_path, data)

def get_list(data,name):
    result_list = []
    for item in data:
        if name in item:
            result_list.append(item[name])
    return result_list

def get_list_for_qu(data,name,instruction):
    result_list = []
    for item in data:
        if name in item:
            result_list.append(instruction+item[name])
    return result_list

def save_json(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data to {file_path}. Error: {e}")



def run(model_name_or_path, finetuned_model_path, result_path, test_data_path_dir, get_model_embedding_batch_size):
    test_batch_size = 1000
    distance_metric ="cosine"
    # load model
    global tokenizer, config, encoder, model, device, n_gpu  #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.debug("device: %s, n_gpu: %s",device, n_gpu)

    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    config = RobertaConfig.from_pretrained(model_name_or_path)
    encoder = RobertaModel.from_pretrained(model_name_or_path)
    model = CodeBertModel(encoder)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)  

    model_to_load = model.module if hasattr(model, 'module') else model 
    model_to_load.load_state_dict(torch.load(finetuned_model_path))
    model.eval()
    model.to(device)
    
    logging.debug("Model and configuration have been successfully loaded!")

    # Run CodeSearchNet's test for each language
    for language in ('python', 'go', 'javascript', 'java', 'php', 'ruby'):
        start_time = time.time()
        print("Evaluating language: {}".format(language))

        #Create a result saving path and a file reading path
        output_cal_path = os.path.join(result_path, "test", language)
        mrr_path = os.path.join(result_path, "test", language, f"{language}_mrr.json")
        data_code_path = "{}/test/{}/{}_data_code_repre.npy".format(result_path, language, language)
        data_docstring_path = "{}/test/{}/{}_data_docstring_repre.npy".format(result_path, language, language)
        test_data_file= "{}/{}/{}/final/jsonl/test/{}_test_0.jsonl".format(test_data_path_dir,language,language, language)
        
        
        if not os.path.exists(output_cal_path):
            os.makedirs(output_cal_path, exist_ok=True)
            print(f"Directory created: {output_cal_path}")
        else:
            print(f"Directory already exists: {output_cal_path}")

        data = read_jsonl(test_data_file)

        # The query and code embedding are vectors and saved
        if os.path.exists(data_code_path) :
            data_code_representation = load_representations(data_code_path)
        else:
            data_code =get_list(data,"code")
            print("Shape of data_code:", np.array(data_code).shape)
            data_code_representation = get_model_embedding(data_code,get_model_embedding_batch_size,'code_inputs', model)
            save_representations(data_code_representation, data_code_path)
            print("Shape of data_code_representation:", np.array(data_code_representation).shape)

            

        if os.path.exists(data_docstring_path):
            data_docstring_representation = load_representations(data_docstring_path)

        else:
            instruction = ("Given a code search query, retrieve relevant passages that answer the query:")

            data_docstring =get_list_for_qu(data,"docstring",instruction)
            print("Shape of data_docstring:", np.array(data_docstring).shape)

            data_docstring_representation = get_model_embedding(data_docstring,get_model_embedding_batch_size,'nl_inputs', model)
            save_representations(data_docstring_representation, data_docstring_path)
            print("Shape of data_docstring_representation:", np.array(data_docstring_representation).shape)

        # Calculate MRR
        mrr_score_value =  compute_evaluation_metrics(data_code_representation,data_docstring_representation,test_batch_size,distance_metric)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to create/check directory: {elapsed_time} seconds")
        mrr_score = {f'{language}_mrr': mrr_score_value,'elapsed_time':elapsed_time}
        save_json(mrr_score, mrr_path)
        print(f"{language} MRR Score: {mrr_score[language + '_mrr']}")



def main():
    parser = argparse.ArgumentParser(description="Run LLM2Vec evaluation.")
    
    parser.add_argument("--model_name_or_path", type=str, required=True, help="The model checkpoint for weights initialization.")
    parser.add_argument("--finetuned_model_path", type=str, required=True, help="The model checkpoint for weights initialization.")
    parser.add_argument("--result_path", type=str,  default="",help="The path to save the results.")
    parser.add_argument("--test_data_language_path_dir", type=str, default="", required=True, help="The directory containing the test data.")
    parser.add_argument("--embedding_batch_size", type=int, default=500, help="Batch size for getting model embeddings.")

    args = parser.parse_args()



    # Call run function
    run(args.model_name_or_path, args.finetuned_model_path, args.result_path, args.test_data_language_path_dir, args.embedding_batch_size)

if __name__ == "__main__":
    main()
