import sys
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import time
from tqdm import tqdm

from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

from typing import Optional, List, Dict, Any, NamedTuple, Iterable, Tuple
from more_itertools import chunked, flatten
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

def Calculate_MAP(sort_lists,eval_file,query_idxs):
    sum = 0
    cnt = 0
    for idx,item in tqdm(zip(query_idxs, sort_lists),total=len(query_idxs)):
        sum += Calculate_AP(item,eval_file,idx)
        cnt += 1
        
    Map = sum / cnt 
    print(f'eval_Map:{Map}')
    return Map

def Calculate_AP(sort_list,data,query_idx):
  
    code_idxs = [item['code-idx'] for item in data if item['query-idx'] == query_idx]
    # print(code_idxs)
    # To sort code_idxs ascending order (otherwise the denominator will be zero)
    code_idxs = sorted(code_idxs)
    # print(code_idxs)
    
    # Find the rank of code-idx in the list and invert it
    ranks = []
    inverse_ranks = []
    
    for code_idx in code_idxs:
        try: 
            rank = sort_list.index(code_idx)+1
            if rank <= 1000:
                ranks.append(rank)
            else:
                ranks.append(0)
        except ValueError:
            ranks.append(0)
    ranks = sorted(ranks) #Ascending sort
    i=1
    for j in range(len(ranks)):
        if not ranks[j]==0:
            inverse_ranks.append((j+1)/ranks[j])
            i+=1
        else:
            inverse_ranks.append(0)
    # print(f'ranks:{ranks}')
        
    AP = sum(inverse_ranks) / len(inverse_ranks)
    # print(f'The {query_idx}th query MrRR is {MrRR}')
    return AP

# sort_lists are code_idxs in descending order of relevance
def CalculateMRR(sort_lists,eval_file,query_idxs):

    data = eval_file
    ranks = []
    inverse_ranks = []
    for idx,item in zip(query_idxs, sort_lists):
        # Find the correct code for a given query-idx
        code_idxs = [item['code-idx'] for item in data if item['query-idx'] == idx] 
        # print(f'code_idxs:{code_idxs}')
        rank_i = []
        for code_idx in code_idxs:
            try:
                # 
                rank = item.index(code_idx)+1
                if rank <= 1000:
                    rank_i.append(rank)
                else:
                    rank_i.append(0) 
            except ValueError:
                rank_i.append(0)

        rank_x = [num for num in rank_i if num > 0]
        rank_min = 0
        if rank_x:
            rank_min = min(rank_x)
        ranks.append(rank_min)
        # print(f'ranks:{ranks}')
    for rank in ranks:
        if not rank == 0:
            inverse_ranks.append(1/rank)
        else:
            inverse_ranks.append(0)
    MRR = sum(inverse_ranks) / len(inverse_ranks)
    print(f'eval_mrr:{MRR}')
    return MRR
     

def load_representations(file_path: str) -> np.ndarray:
    return np.load(file_path)

def get_model_embedding(date_p,embedding_batch_size):
    batch_chunked_data = chunked(date_p, embedding_batch_size) 
    print(f"Each batch includes {embedding_batch_size} data points")

    embeddings_list = []
    sum_batch_number = 0
    total_batches = len(date_p) // embedding_batch_size + (1 if len(date_p) % embedding_batch_size != 0 else 0)
    for batch_data in tqdm(batch_chunked_data, total=total_batches, desc="Processing batches"):
        batch_data_representation = l2v.encode(batch_data)
        embeddings_list.append(batch_data_representation)
        sum_batch_number = sum_batch_number +1

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
            result_list.append([instruction,item[name]])
    return result_list

def save_json(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data to {file_path}. Error: {e}")




def run(path, result_path, test_data_path, get_model_embedding_batch_size):
    start_time = time.time()
    test_batch_size = 1000
    distance_metric ="cosine"

    output_cal_path = "{}".format(result_path)
    map_path = "{}/map.json".format(result_path)
    data_code_path = "{}/data_code_repre.npy".format(result_path)
    data_docstring_path = "{}/data_docstring_repre.npy".format(result_path)


    query_path = os.path.join(test_data_path, "query.json")
    codebase_path = os.path.join(test_data_path, "final_augment_codebase.json")
    true_pair_file_path = os.path.join(test_data_path, "final_augment_query_code_pairs_for_search.json")


    # load model
    global l2v
    l2v = LLM2Vec.from_pretrained(
        path,
        enable_bidirectional=False,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        pooling_mode="mean",
    )

    if not os.path.exists(output_cal_path):
        os.makedirs(output_cal_path, exist_ok=True)
        print(f"Directory created: {output_cal_path}")
    else:
        print(f"Directory already exists: {output_cal_path}")

    logger.info("loading data...")
    with open(query_path, 'r') as f:
        query_dataset = json.load(f)
    with open(codebase_path,'r') as f:
        code_dataset = json.load(f)
    with open(true_pair_file_path,'r') as f:
        true_pair_file = json.load(f)


    if os.path.exists(data_docstring_path):
        nl_vecs = load_representations(data_docstring_path)
    else:
        instruction = ("Given a code search query, retrieve relevant passages that answer the query:")
        data_docstring =get_list_for_qu(query_dataset,"query",instruction)
        print("Shape of data_docstring:", np.array(data_docstring).shape)
        nl_vecs = get_model_embedding(data_docstring,get_model_embedding_batch_size)
        save_representations(nl_vecs, data_docstring_path)
        print("Shape of data_docstring_representation:", np.array(nl_vecs).shape)
    
    if os.path.exists(data_code_path) :
        code_vecs = load_representations(data_code_path)
    else:
        data_code =get_list(code_dataset , "code")
        print("Shape of data_code:", np.array(data_code).shape)
        code_vecs = get_model_embedding(data_code,get_model_embedding_batch_size)
        save_representations(code_vecs, data_code_path)
        print("Shape of data_code_representation:", np.array(code_vecs).shape)


    # Sort the dictionary's items() and then extract values()
    sorted_code_vecs = [code_vecs[item['code-idx']] for item in code_dataset]
    sorted_nl_vecs = [nl_vecs[item['query-idx']] for item in query_dataset]

    # Converts the sorted embedded list to a NumPy array
    code_vecs_np = np.array(sorted_code_vecs)
    nl_vecs_np = np.array(sorted_nl_vecs)
    logger.info("embedding done and saved!")
    scores = np.matmul(nl_vecs_np,code_vecs_np.T)#Matrix Matrix multiplication computes the product of two matrices
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    logger.info("sort done!")
    # sort_ids is not idx, but the rank projected by code
    sort_idxs = []
    for sort_id in tqdm(sort_ids): 
        sort_idx = []
        for i in sort_id[:1000]: #Get the first 1000 of the sorted index.
            sort_idx.append(code_dataset[i]['code-idx']) 
        sort_idxs.append(sort_idx) 
    
    # To get all query-idxs corresponding to sort_ids
    query_idxs = []
    for example in tqdm(query_dataset):
        query_idxs.append(example['query-idx'])
        
    # Calculate the MAP and MRR
    logger.info("calculating Map...")
    Map = Calculate_MAP(sort_idxs,true_pair_file,query_idxs)
    mrr = CalculateMRR(sort_idxs,true_pair_file,query_idxs)
    end_time = time.time()
    elapsed_time = end_time - start_time

    result = {
        "eval_mrr":float(mrr),
        "eval_Map":float(Map),
        "time":float(Map),
    }

    save_json(result, map_path)
        
    # print(f"Ruby MRR Score: {mrr_score[f'{language}_mrr']}")
    # print(f"Map Score: { result['eval_Map']}")
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],3)))


def main():
    parser = argparse.ArgumentParser(description="Run LLM2Vec evaluation.")
    
    parser.add_argument("--model_name_or_path", type=str, required=True, help="The model checkpoint for weights initialization.")
    parser.add_argument("--result_path", type=str,  default="",help="The path to save the results.")
    parser.add_argument("--test_data_path_dir", type=str, default="", required=True, help="The directory containing the test data.")
    parser.add_argument("--embedding_batch_size", type=int, default=500, help="Batch size for getting model embeddings.")

    args = parser.parse_args()

    # Call run function
    run(args.model_name_or_path, args.result_path, args.test_data_path_dir, args.embedding_batch_size)

if __name__ == "__main__":
    main()