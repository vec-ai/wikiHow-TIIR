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
from PIL import Image
import io
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

logger = logging.getLogger()
import random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import load_model
import glob
import pickle
import os
import re


class WikihowInterleavedData:
    def __init__(self, instance):
        self.vl_data_query = instance['vl_data_query'] if 'vl_data_query' in instance else None
        self.vl_data_doc = instance['vl_data_doc'] if 'vl_data_doc' in instance else None
        self.image_query = instance['image_query'] if 'image_query' in instance else None
        self.image_doc = instance['image_doc'] if 'image_doc' in instance else None
        self.index = instance['index'] if 'index' in instance else None



class WikihowInterleaved(Dataset):
    def __init__(self, args, preprocess_fn, vl_chat_processor, qrels_path, doc_path, query_path, doc_image_root, only_text, only_image):

        self.preprocess_fn = preprocess_fn
        self.vl_chat_processor=vl_chat_processor

        self.doc_image_root = doc_image_root
        self.qrels_path = qrels_path
        self.doc_path = doc_path
        self.query_path = query_path

        self.only_text = only_text
        self.only_image = only_image

        self.qrels = []
        self.docs = {}
        self.queries = {}

        self.init()


    def init(self):
        ids = []
        with open(self.qrels_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.qrels.append((data['did'],data['qid']))
 

        with open(self.doc_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.docs[data['id']] = {'data':data['data']}
                self.docs[data['id']]['vl_data'] = self.create_deepseek_conversations(data['data'])
                ids.append(data['id'])

        with open(self.query_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.queries[data['qid']] = {'data':data['data']}
                self.queries[data['qid']]['vl_data'] = self.create_deepseek_conversations(data['data'])

        print(f"data_len:{len(self.qrels)}\n")


    def __len__(self):
        return len(self.qrels)


    def encode_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)
        return {'img': img}


    def encode_imgs(self, img_paths):
        images = []
        for ipath in img_paths:
            images += [self.encode_img(ipath)["img"]] 
        return images


    def create_deepseek_conversations(self, data):
        conversations = []
        text = None
        images = []
        
        for d in data:
            if d.endswith('.png'):
                if self.only_text:
                    continue
                images.append(d)
                if text is None:
                    text = "<image_placeholder> "
                else:
                    text += "\n <image_placeholder>"
            else:
                if self.only_image:
                    continue
                if text is None:
                    text = d
                else:
                    text += d
        conversations.append({
                "role": "User",
                "content": text,
                "images": images 
            })
        conversations.append({
                "role": "Assistant",
                "content": "<｜end▁of▁sentence｜>"
            })
        return conversations,images


    def Collector(self, batch):
        vl_data_query = []
        vl_data_doc = []
        image_query = []
        image_doc = []
        index = []
        processed_batch = {}
        for qid, example in enumerate(batch):
            if example.vl_data_query is not None:
                vl_data_query.append(example.vl_data_query)
                image_query.append(example.image_query)
            
            vl_data_doc.append(example.vl_data_doc)  
            image_doc.append(example.image_doc)
            index.append(example.index)

        if len(vl_data_query) > 0 :
            processed_batch['vl_data_query'] = self.vl_chat_processor(
                conversations=vl_data_query,
                images=image_query,
                force_batchify=True
            )
        
        processed_batch['vl_data_doc'] = self.vl_chat_processor(
            conversations=vl_data_doc,
            images=image_doc,
            force_batchify=True
        )
        processed_batch['index'] = index

        return processed_batch


    def __getitem__(self, index):
        data_id = self.qrels[index][1]
        if data_id is None:
            query,qimages =  None,None
            data_id = self.qrels[index][0]
        else:
            query,qimages = self.queries[data_id]['vl_data']
        
        doc,dimages = self.docs[data_id]['vl_data']
        instance = {}
        if query is not None:
            instance['vl_data_query'] = query
            instance['image_query'] = self.encode_imgs(qimages)
        
        instance['vl_data_doc'] = doc
        instance['image_doc'] = self.encode_imgs(dimages)
        instance['index'] = index

        return WikihowInterleavedData(instance)



def gen_embeddings(model, valid_reader, outpath, accelerator):
    model.eval()
    all_query_embeddings = []
    all_candidate_embeddings = []
    all_index = []
    total_batches = len(valid_reader)
    for step, batch in tqdm(enumerate(valid_reader), total=total_batches, disable=not accelerator.is_main_process):
        with torch.no_grad():
            if "vl_data_query" in batch and batch["vl_data_query"] is not None:
                query_embedding = model.get_embeds(**batch["vl_data_query"])  # Process queries
                query_embedding = F.normalize(query_embedding, dim=-1).cpu()
                all_query_embeddings.append(torch.tensor(query_embedding,dtype = torch.float32))
            candidate_embeddings = model.get_embeds(**batch["vl_data_doc"])  # Process queries
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1).cpu()
            all_candidate_embeddings.append(torch.tensor(candidate_embeddings,dtype = torch.float32))
            all_index.append(torch.tensor(batch["index"]))
    all_query_embeddings = torch.cat(all_query_embeddings, dim=0).numpy()
    all_candidate_embeddings = torch.cat(all_candidate_embeddings, dim=0).numpy()
    all_index = torch.cat(all_index, dim=0).numpy()
    with open(outpath[0], 'wb') as fout:
        pickle.dump((all_index, all_query_embeddings), fout)
    with open(outpath[1], 'wb') as fout:
        pickle.dump((all_index, all_candidate_embeddings), fout)


def data2embeddings(data, accelerator, model, output_path):
    sampler = SequentialSampler(data)
    reader = DataLoader(dataset=data, sampler=sampler, num_workers=args.num_workers,
                                batch_size=args.batch_size, collate_fn=data.Collector)
    model,reader = accelerator.prepare(model,reader)
    model = model.to(dtype=torch.bfloat16)
    gen_embeddings(model, reader, output_path, accelerator)


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
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pooler_type", type=str, default=None, help='pooler_type in first, last, mean, weightedmean')
    parser.add_argument("--do_norm",type=str2bool,default=False)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--mixed_precision", type=str)
    parser.add_argument("--use_lora", type=str2bool,default=True)
    parser.add_argument("--do_lora", type=str2bool,default=False)
    parser.add_argument("--image_token_num", type=int, default=576)
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--doc_path", type=str)
    parser.add_argument("--query_path", type=str)
    parser.add_argument("--doc_image_root", type=str)
    parser.add_argument("--bf16_model", type=str2bool,default=True)
    parser.add_argument("--is_matry", type=str2bool,default=True)
    parser.add_argument("--maxtoken", type=int,default=None)
    parser.add_argument("--only_text", type=str2bool,default=False)
    parser.add_argument("--only_image", type=str2bool,default=False)
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
    vl_chat_processor, model, image_processor = load_model(args)
    if args.maxtoken is not None:
        vl_chat_processor.max_len = args.maxtoken
    data = WikihowInterleaved(args, image_processor, vl_chat_processor, args.qrels_path, args.doc_path, args.query_path, args.doc_image_root, args.only_text, args.only_image)
    output = [
        os.path.join(args.out_path, f'query_embedding-{accelerator.process_index}.pkl'),
        os.path.join(args.out_path, f'doc_embedding-{accelerator.process_index}.pkl')
    ]
    data2embeddings(data,accelerator,model,output)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.only_text:
            merge_pkl_files(args.out_path, 'query_embedding-*.pkl', os.path.join(args.out_path, f'query_embedding_text.pkl'),accelerator)
            merge_pkl_files(args.out_path, 'doc_embedding-*.pkl', os.path.join(args.out_path, f'doc_embedding_text.pkl'),accelerator)
        else:
            merge_pkl_files(args.out_path, 'query_embedding-*.pkl', os.path.join(args.out_path, f'query_embedding_{args.image_token_num}.pkl'),accelerator)
            merge_pkl_files(args.out_path, 'doc_embedding-*.pkl', os.path.join(args.out_path, f'doc_embedding_{args.image_token_num}.pkl'),accelerator)