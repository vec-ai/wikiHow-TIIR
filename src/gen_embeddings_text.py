import json
import logging
import sys
import base64
import os
from typing import Optional
import numpy as np
from torch import nn
from torch.nn import LayerNorm
from tqdm import tqdm
import torch
import argparse
import os.path as op
import time
import pickle
import math
import base64
import io
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from torch import Tensor
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import random
from PIL import Image
from PIL import ImageFile
import glob
import pickle
import re

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger()


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class WikihowInterleavedData:
    def __init__(self, instance):
        self.query = instance['query'] if 'query' in instance else None
        self.doc = instance['doc'] if 'doc' in instance else None
        self.index = instance['index'] if 'index' in instance else None


class WikihowInterleaved(Dataset):
    def __init__(self, args, tokenizer, qrels_path, doc_path, query_path):
        self.args = args
        self.tokenizer = tokenizer
        self.qrels_path = qrels_path
        self.doc_path = doc_path
        self.query_path = query_path
        self.qrels = []
        self.docs = {}
        self.queries = {}
        self.dataids = {}
        self.gte_inst_prompt = 'Given a query, retrieve relevant wikiHow document that answer the query'
        self.bge_inst_prompt = 'Represent this query for searching relevant wikiHow passages:'
        self.init()


    def init(self):
        with open(self.qrels_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.qrels.append((data['did'],data['qid']))
                self.dataids[data['did']] = data['qid']
        with open(self.doc_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                doc = []
                for d in data['data']:
                    if not d.endswith('.png'):
                        doc.append(d)
                self.docs[data['id']] = '\n'.join(doc)
                if self.args.model == 'gte-large':
                    self.docs[data['id']] = f"passage:{self.docs[data['id']]}"
        with open(self.query_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                query = []
                for d in data['data']:
                    if not d.endswith('.png'):
                        query.append(d)
                id_name = 'id' if 'id' in data else 'qid'
                self.queries[data[id_name]] = '\n'.join(query)
                if self.args.model == 'gte-inst':
                    self.queries[data[id_name]] = get_detailed_instruct(self.gte_inst_prompt, self.queries[data[id_name]])
                elif self.args.model == 'gte-large':
                    self.queries[data[id_name]] = f"query:{self.queries[data[id_name]]}"
                elif self.args.model == 'bge-large':
                    self.queries[data[id_name]] = f"{self.bge_inst_prompt} {self.queries[data[id_name]]}"
        print(f"data_len:{len(self.qrels)}\n")


    def __len__(self):
        return len(self.qrels)

        
    def Collector(self, batch):
        query = []
        doc = []
        processed_batch = {}
        for qid, example in enumerate(batch):
            if example.query is not None:
                query.append(example.query)
            doc.append(example.doc)
        if len(query) > 0:
            processed_batch['query'] = self.tokenizer(query, max_length=self.args.maxtoken, padding=True, truncation=True, return_tensors='pt')
        processed_batch['doc'] = self.tokenizer(doc, max_length=self.args.maxtoken, padding=True, truncation=True, return_tensors='pt')
        processed_batch['index'] = torch.tensor([example.index for example in batch])
        return processed_batch


    def __getitem__(self, index):
        data_id = self.qrels[index][1]
        if data_id is None:
            query =  None
            data_id = self.qrels[index][0]
        else:
            query= self.queries[data_id]
        doc= self.docs[data_id]
        instance = {}
        if query is not None:
            instance['query'] = query
        instance['doc'] = doc
        instance['index'] = index
        return WikihowInterleavedData(instance)


def get_embed(args, model, inputs, isQuery=False):
    if args.model == 'gte-inst':
        outputs = model(**inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu()
    elif args.model == 'e5-large':
        outputs = model(**inputs)
        embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu()
    elif args.model == 'bge-large':
        model_output = model(**inputs)
        # Perform pooling. In this case, cls pooling.
        embeddings = model_output[0][:, 0]
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu()
    else:
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, dim=-1).cpu()
    return embeddings


def gen_embeddings(model, valid_reader, outpath, accelerator, args):
    model.eval()
    all_query_embeddings = []
    all_candidate_embeddings = []
    all_index = []
    total_batches = len(valid_reader)
    for step, batch in tqdm(enumerate(valid_reader), total=total_batches, disable=not accelerator.is_main_process):
        with torch.no_grad():
            if "query" in batch and batch["query"] is not None:
                query_embedding = get_embed(args, model, batch["query"], isQuery=True)
                all_query_embeddings.append(torch.tensor(query_embedding,dtype = torch.float32))
            doc_embedding = get_embed(args, model, batch["doc"])
            all_candidate_embeddings.append(torch.tensor(doc_embedding,dtype = torch.float32))
            all_index.append(batch["index"].cpu())
    all_query_embeddings = torch.cat(all_query_embeddings, dim=0).numpy()
    all_candidate_embeddings = torch.cat(all_candidate_embeddings, dim=0).numpy()
    all_index = torch.cat(all_index, dim=0).numpy()
    with open(outpath[0], 'wb') as fout:
        pickle.dump((all_index, all_query_embeddings), fout)
    with open(outpath[1], 'wb') as fout:
        pickle.dump((all_index, all_candidate_embeddings), fout)


def data2embeddings(data, accelerator, model, output_path, args):
    sampler = SequentialSampler(data)
    reader = DataLoader(dataset=data, sampler=sampler, num_workers=args.num_workers,
                                batch_size=args.batch_size, collate_fn=data.Collector)
    model,reader = accelerator.prepare(model,reader)
    model = model.to(dtype=torch.bfloat16)
    gen_embeddings(model, reader, output_path, accelerator, args)


def merge_pkl_files(input_path, file_pattern, output_path, accelerator):
    """
    Merges all .pkl files matching the file_pattern in input_path into one output .pkl file.
    
    Parameters:
    input_path (str): The folder path where .pkl files are located.
    file_pattern (str): The pattern to match .pkl files.
    output_path (str): The path including filename where the merged .pkl file will be saved.
    """
    full_file_pattern = f"{input_path}/{file_pattern}"
    file_paths = sorted(glob.glob(full_file_pattern), key=lambda x: int(re.findall(r'-(\d+).pkl', x)[0]))
    assert len(file_paths) == accelerator.num_processes
    all_indices = []
    all_embeddings = []
    out_indices = []
    out_embeddings = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            indices, embeddings = pickle.load(f)
            all_indices.extend(indices)
            all_embeddings.extend(embeddings)
    data_dict = {}
    for i in tqdm(range(len(all_embeddings))):
        if all_indices[i] in data_dict:
            continue
        data_dict[all_indices[i]] = True
        out_indices.append(all_indices[i])
        out_embeddings.append(all_embeddings[i])
    with open(output_path, 'wb') as f:
        pickle.dump((out_indices, out_embeddings), f)
    for file_path in file_paths:
        os.remove(file_path)
    
    print(f"All embeddings have been merged into {output_path} and original files have been removed.")

def str2bool(value):
    """将字符串转换为布尔值"""
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--mixed_precision", type=str)
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--doc_path", type=str)
    parser.add_argument("--query_path", type=str)
    parser.add_argument("--bf16_model", type=str2bool,default=False)
    parser.add_argument("--maxtoken", type=int,default=None)
    parser.add_argument("--model", type=str,default=None)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision, device_placement=True)

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'embedding.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    data = WikihowInterleaved(args, tokenizer, args.qrels_path, args.doc_path, args.query_path)
    output = [
        os.path.join(args.out_path, f'query_embedding-{accelerator.process_index}.pkl'),
        os.path.join(args.out_path, f'doc_embedding-{accelerator.process_index}.pkl')
    ]
    data2embeddings(data,accelerator,model,output, args)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merge_pkl_files(args.out_path, 'query_embedding-*.pkl', os.path.join(args.out_path, f'query_embedding_{args.model}.pkl'),accelerator)
        merge_pkl_files(args.out_path, 'doc_embedding-*.pkl', os.path.join(args.out_path, f'doc_embedding_{args.model}.pkl'),accelerator)