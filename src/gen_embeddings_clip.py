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
from PIL import ImageFile
from utils import load_model
import glob
import pickle
import random
import re
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger()


class WikihowInterleavedData:
    def __init__(self, instance):
        self.text_query = instance['text_query'] if 'text_query' in instance else None
        self.text_doc = instance['text_doc'] if 'text_doc' in instance else None
        self.image_query = instance['image_query'] if 'image_query' in instance else None
        self.image_doc = instance['image_doc'] if 'image_doc' in instance else None
        self.index = instance['index'] if 'index' in instance else None
        self.qimage_index = instance['qimage_index'] if 'qimage_index' in instance else None
        self.dimage_index = instance['dimage_index'] if 'dimage_index' in instance else None


class WikihowInterleaved(Dataset):
    def __init__(self, args, tokenizer, processor, qrels_path, doc_path, query_path):
        self.args = args
        self.processor = processor
        self.tokenizer = tokenizer
        self.qrels_path = qrels_path
        self.doc_path = doc_path
        self.query_path = query_path

        self.qrels = []
        self.docs = {}
        self.queries = {}
        self.init()


    def init(self):
        with open(self.qrels_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.qrels.append((data['did'],data['qid']))
        with open(self.doc_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.docs[data['id']] = {'data':data['data']}
                self.docs[data['id']]['vl_data'] = self.prepare_clip_data(data['data'])
        with open(self.query_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.queries[data['qid']] = {'data':data['data']}
                self.queries[data['qid']]['vl_data'] = self.prepare_clip_data(data['data'])

        print(f"data_len:{len(self.qrels)}\n")


    def __len__(self):
        return len(self.qrels)

    
    def concatenate_images_horizontally(self, image_list):
        if not image_list:
            raise ValueError("Image list is empty")
        images = [img.convert('RGB') for img in image_list]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
        return new_img


    def encode_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return {'img': img}


    def encode_imgs(self, img_paths):
        images = []
        for ipath in img_paths:
            if isinstance(ipath, str):
                images.append(self.encode_img(ipath)["img"])
            else:
                images.append(self.concatenate_images_horizontally([self.encode_img(i)["img"] for i in ipath]))
        if self.processor:
            return self.processor(images, return_tensors="pt")
        return images


    def prepare_clip_data(self, data):
        text = ""
        images = []
        
        for d in data:
            if d.endswith('.png'):
                images.append(d)
            else:
                text += f" {d}"
        
        return text,images


    def Collector(self, batch):
        text_query = []
        text_doc = []
        image_query = []
        image_doc = []
        qimage_index = []
        dimage_index = []
        processed_batch = {}
        for qid, example in enumerate(batch):
            if example.text_query is not None:
                text_query.append(example.text_query)
                if self.args.image_type == 'mean':
                    image_query.extend(example.image_query)
                else:
                    image_query.append(example.image_query)
                if qid == 0:
                    qimage_index.append([0,len(example.image_query)])
                else:
                    qimage_index.append([qimage_index[-1][1],qimage_index[-1][1]+len(example.image_query)])
            
            text_doc.append(example.text_doc)  
            if self.args.image_type == 'mean':
                image_doc.extend(example.image_doc)
            else:
                image_doc.append(example.image_doc)
            if qid == 0:
                dimage_index.append([0,len(example.image_doc)])
            else:
                dimage_index.append([dimage_index[-1][1],dimage_index[-1][1]+len(example.image_doc)])
        if len(text_query) > 0:
            qimage_index = torch.tensor(qimage_index)
        else:
            qimage_index = None
        dimage_index = torch.tensor(dimage_index)

        if len(text_query) > 0:
            if 'jina' in self.args.model:
                processed_batch['text_query'] = text_query
            else:
                processed_batch['text_query'] = self.tokenizer(text_query, padding=True, truncation=True, return_tensors="pt")
            processed_batch['image_query'] = self.encode_imgs(image_query)
            processed_batch['qimage_index'] = qimage_index
        
        if 'jina' in self.args.model:
            processed_batch['text_doc'] = text_doc
        else:
            processed_batch['text_doc'] = self.tokenizer(text_doc, padding=True, truncation=True, return_tensors="pt")
        processed_batch['image_doc'] = self.encode_imgs(image_doc)
        processed_batch['index'] = torch.tensor([example.index for example in batch])
        processed_batch['dimage_index'] = dimage_index
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
            instance['text_query'] = query
            instance['image_query'] = qimages
        
        instance['text_doc'] = doc
        instance['image_doc'] = dimages
        instance['index'] = index

        return WikihowInterleavedData(instance)


def encode(args,model,text,image,isQuery):
    if args.model == 'jina':
        if isQuery:
            text_embeding = model.encode_text(
                text, task='retrieval.query'
            )
        else:
            text_embeding = model.encode_text(
                text
            )
        text_embeding = torch.tensor(text_embeding)
        image_embeding = model.encode_image(
            image
        )
        image_embeding = torch.tensor(image_embeding)
    else:
        text_embeding = model.get_text_features(**text)
        image_embeding = model.get_image_features(**image)
        text_embeding /= text_embeding.norm(dim=-1, keepdim=True)
        image_embeding /= image_embeding.norm(dim=-1, keepdim=True)
        
    return text_embeding,image_embeding


def gen_embeddings(model, valid_reader, outpath, accelerator, args):
    model.eval()
    all_query_embeddings = []
    all_candidate_embeddings = []
    all_index = []
    total_batches = len(valid_reader)
    for step, batch in tqdm(enumerate(valid_reader), total=total_batches, disable=not accelerator.is_main_process):
        with torch.no_grad():
            if "text_query" in batch and batch["text_query"] is not None:
                query_text_embeding,qimage_embeding = encode(args,model,batch["text_query"],batch["image_query"],True)
                if qimage_embeding.shape[0] == query_text_embeding.shape[0]:
                    query_image_embeding = qimage_embeding
                else:
                    query_image_embeding = []
                    for qindex in batch["qimage_index"]:
                        query_image_embeding.append(qimage_embeding[qindex[0]:qindex[1]].mean(dim=0))
                    query_image_embeding = torch.stack(query_image_embeding)
                if args.it_type == 'NAddNorm' or args.it_type == 'NCNorm' or args.it_type == 'NPNorm':
                    query_text_embeding = F.normalize(query_text_embeding, dim=-1)
                    query_image_embeding = F.normalize(query_image_embeding, dim=-1)
                if args.it_type == 'addNorm' or args.it_type == 'NAddNorm':
                    query_embedding = query_text_embeding+query_image_embeding
                elif args.it_type == 'concatNorm' or args.it_type == 'NCNorm':
                    query_embedding = torch.cat([query_text_embeding,query_image_embeding],dim=-1)
                elif args.it_type == 'productNorm' or args.it_type == 'NPNorm':
                    query_embedding = query_text_embeding*query_image_embeding

                query_embedding = F.normalize(query_embedding, dim=-1).cpu()
                all_query_embeddings.append(torch.tensor(query_embedding,dtype = torch.float32))
            
            doc_text_embeding,dimage_embeding = encode(args,model,batch["text_doc"],batch["image_doc"],False)
            if dimage_embeding.shape[0] == doc_text_embeding.shape[0]:
                candidate_embeddings = dimage_embeding
            else:
                candidate_embeddings = []
                for dindex in batch["dimage_index"]:
                    candidate_embeddings.append(dimage_embeding[dindex[0]:dindex[1]].mean(dim=0))
                candidate_embeddings = torch.stack(candidate_embeddings)
            if args.it_type == 'NAddNorm' or args.it_type == 'NCNorm' or args.it_type == 'NPNorm':
                doc_text_embeding = F.normalize(doc_text_embeding, dim=-1)
                candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
            if args.it_type == 'addNorm' or args.it_type == 'NAddNorm':
                candidate_embeddings = doc_text_embeding+candidate_embeddings
            elif args.it_type == 'concatNorm' or args.it_type == 'NCNorm':
                candidate_embeddings = torch.cat([doc_text_embeding,candidate_embeddings],dim=-1)
            elif args.it_type == 'productNorm' or args.it_type == 'NPNorm':
                candidate_embeddings = doc_text_embeding*candidate_embeddings
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1).cpu()

            all_candidate_embeddings.append(torch.tensor(candidate_embeddings,dtype = torch.float32))
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
    model = model.to(dtype=torch.float16)
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
    parser.add_argument("--it_type", type=str,default=None, help='addNorm,NAddNorm,concatNorm,NCNorm,productNorm,NPNorm')
    parser.add_argument("--image_type", type=str,default=None, help='mean,concat')
    parser.add_argument("--model", type=str,default=None, help='jina,openai')
    parser.add_argument("--eva_model_name", type=str,default=None, help='jina,openai')
    

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
    processor, model, tokenizer = load_model(args)
    data = WikihowInterleaved(args, tokenizer, processor, args.qrels_path, args.doc_path, args.query_path)
    output = [
        os.path.join(args.out_path, f'query_embedding-{accelerator.process_index}.pkl'),
        os.path.join(args.out_path, f'doc_embedding-{accelerator.process_index}.pkl')
    ]
    data2embeddings(data,accelerator,model,output, args)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merge_pkl_files(args.out_path, 'query_embedding-*.pkl', os.path.join(args.out_path, f'query_embedding_{args.it_type}_{args.image_type}.pkl'),accelerator)
        merge_pkl_files(args.out_path, 'doc_embedding-*.pkl', os.path.join(args.out_path, f'doc_embedding_{args.it_type}_{args.image_type}.pkl'),accelerator)