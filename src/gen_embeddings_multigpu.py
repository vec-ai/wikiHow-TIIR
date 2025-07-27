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
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import random
from PIL import Image
from PIL import ImageFile
import glob
import pickle
import re
from modeling_nvmmembed import NVMMEmbedModel
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoModel
from gme_inference import GmeQwen2VL
from visual_bge.modeling import Visualized_BGE
logger = logging.getLogger()
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WikihowInterleavedData:
    def __init__(self, instance):
        self.text_query = instance['text_query'] if 'text_query' in instance else None
        self.text_doc = instance['text_doc'] if 'text_doc' in instance else None
        self.image_query = instance['image_query'] if 'image_query' in instance else None
        self.image_doc = instance['image_doc'] if 'image_doc' in instance else None
        self.index = instance['index'] if 'index' in instance else None


class WikihowInterleaved(Dataset):
    def __init__(self, args, model, processor, qrels_path, doc_path, query_path, isQuery):

        self.args = args
        self.processor = processor
        self.qrels_path = qrels_path
        self.doc_path = doc_path
        self.query_path = query_path
        self.model = model

        self.qrels = []
        self.docs = {}
        self.queries = {}

        self.isQuery = isQuery

        self.init()
        self.llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
        self.e5v_qprompt = self.llama3_template.format('<image>\n<sent>\nSummary above query in one word: ')
        self.e5v_dprompt = self.llama3_template.format('<image>\n<sent>\nSummary above tutorial in one word: ')
        self.e5v_text_prompt = self.llama3_template.format('<sent>\nSummary above sentence in one word: ')
        self.e5v_text_qprompt = self.llama3_template.format('<sent>\nSummary above query in one word: ')
        self.e5v_text_dprompt = self.llama3_template.format('<sent>\nSummary above tutorial in one word: ')
        

    def init(self):
        with open(self.qrels_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                if self.isQuery and data['qid'] is not None:
                    self.qrels.append(data['qid'])
                elif not self.isQuery and data['did'] is not None:
                    self.qrels.append(data['did'])
        if self.isQuery:
            with open(self.query_path,'r') as f:
                for line in tqdm(f):
                    data = json.loads(line)
                    self.queries[data['qid']] = {'data':data['data']}
                    self.queries[data['qid']]['vl_data'] = self.prepare_clip_data(data['data'])
        else:
            with open(self.doc_path,'r') as f:
                for line in tqdm(f):
                    data = json.loads(line)
                    self.docs[data['id']] = {'data':data['data']}
                    self.docs[data['id']]['vl_data'] = self.prepare_clip_data(data['data'])
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
        if isinstance(img_paths, list) and len(img_paths)==1:
            img_paths = img_paths[0]
        if isinstance(img_paths, str):
            images = self.encode_img(img_paths)["img"]
        else:
            images = self.concatenate_images_horizontally([self.encode_img(i)["img"] for i in img_paths])
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


    def get_batch_data(self, text, image, isQuery=True):
        if self.args.model == 'mmembed':
            if self.args.task_type == 'only_text':
                return {'txt':text}
            return {'txt':text,"img":self.encode_imgs(image)}
        elif self.args.model == 'e5-v':
            if self.args.task_type == 'only_text':
                if isQuery:
                    return (self.e5v_text_qprompt.replace('<sent>', text), None)
                else:
                    return (self.e5v_text_dprompt.replace('<sent>', text), None)
            if isQuery:
                return (self.e5v_qprompt.replace('<sent>', text), self.encode_imgs(image))
            else:
                return (self.e5v_dprompt.replace('<sent>', text), self.encode_imgs(image))
        elif self.args.model == 'gme':
            if self.args.task_type == 'only_text':
                return (text, None)
            return (text,self.encode_imgs(image))
        elif self.args.model == 'bge':
            if self.args.task_type == 'only_text':
                return (text, None)
            return (text,self.encode_imgs(image))
        elif self.args.model == 'jina':
            if self.args.task_type == 'only_text':
                return (text, None)
            return (text,self.encode_imgs(image))
    

    def prepare_batch_data(self, data, isQuery=True):
        if self.args.model == 'mmembed':
            return (data, None)
        elif self.args.model == 'e5-v':
            e5v_inputs = [d[0] for d in data]
            e5v_images = [d[1] for d in data]
            if self.args.task_type == 'only_text':
                return self.processor(e5v_inputs, return_tensors="pt", padding=True)
            return self.processor(e5v_inputs, e5v_images, return_tensors="pt", padding=True)
        elif self.args.model == 'gme':
            gme_text = [d[0] for d in data]
            gme_image = [d[1] for d in data]
            return (gme_text,gme_image)
        elif self.args.model == 'bge':
            bge_text = [d[0] for d in data]
            if self.args.task_type == 'only_text':
                return (model.tokenizer(bge_text, return_tensors="pt", padding=True, truncation=True, max_length = 512),None)
            bge_text = model.tokenizer(bge_text, return_tensors="pt", padding=True, truncation=True, max_length = 512-196)
            bge_image = [self.model.preprocess_val(d[1]).unsqueeze(0) for d in data]
            bge_image = torch.cat(bge_image,dim=0)
            return (bge_text,bge_image)
        elif self.args.model == 'jina':
            jina_text = [d[0] for d in data]
            jina_image = [d[1] for d in data]
            if self.args.task_type == 'only_text':
                return (jina_text,None)
            return (jina_text,jina_image)
        


    def Collector(self, batch):
        query = []
        doc = []
        processed_batch = {}
        for qid, example in enumerate(batch):
            if example.text_query is not None:
                query.append(self.get_batch_data(example.text_query,example.image_query, isQuery=True))
            if example.text_doc is not None:
                doc.append(self.get_batch_data(example.text_doc,example.image_doc, isQuery=False))
        if len(query) > 0:
            processed_batch['query'] = self.prepare_batch_data(query, isQuery=True)        
        if len(doc) > 0:
            processed_batch['doc'] = self.prepare_batch_data(doc, isQuery=False)
        processed_batch['index'] = torch.tensor([example.index for example in batch])
        return processed_batch


    def __getitem__(self, index):
        data_id = self.qrels[index]
        instance = {}
        if self.isQuery:
            query,qimages = self.queries[data_id]['vl_data']
            instance['text_query'] = query
            instance['image_query'] = qimages
        else:
            doc,dimages = self.docs[data_id]['vl_data']
            instance['text_doc'] = doc
            instance['image_doc'] = dimages
        instance['index'] = index
        return WikihowInterleavedData(instance)


def get_embed(args, model, inputs, isQuery=False):
    if args.model == 'mmembed':
        data = inputs[0]
        mmembed_prompt = "Retrieve a Wikihow tutorial that provides an answer to the given query."
        return model.module.encode(data,is_query=isQuery,instruction=mmembed_prompt, max_length = 4096)['hidden_states']
    elif args.model == 'e5-v':
        return model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
    elif args.model == 'gme':
        gme_text = inputs[0]
        gme_image = inputs[1]
        instruction = 'Find a wikiHow tutorial that matches the given query.' if isQuery else None
        if args.task_type == 'only_text':
            return model.get_embeddings(texts=gme_text, instruction=instruction)
        return model.get_embeddings(texts=gme_text, images=gme_image, instruction=instruction)
    elif args.model == 'bge':
        bge_text = inputs[0]
        bge_image = inputs[1]
        if args.task_type == 'only_text':
            return model.encode_text(bge_text)
        return model.module.encode_mm(images=bge_image.to(model.device), texts=bge_text.to(model.device))
    elif args.model == 'jina':
        jina_text = inputs[0]
        jina_image = inputs[1]
        if args.task_type == 'only_text':
            return torch.from_numpy(model.encode_text(jina_text, task='retrieval.query')) if isQuery else torch.from_numpy(model.encode_text(jina_text))


def gen_embeddings(model, query_reader, doc_reader, outpath, accelerator, args):
    if args.model != 'gme':
        model.eval()
    all_query_embeddings = []
    all_candidate_embeddings = []
    all_query_index = []
    all_doc_index = []
    total_query_batches = len(query_reader)
    total_doc_batches = len(doc_reader)
    for step, batch in tqdm(enumerate(query_reader), total=total_query_batches, disable=not accelerator.is_main_process):
        with torch.no_grad():
            query_embedding = get_embed(args, model, batch["query"],isQuery=True)
            query_embedding = F.normalize(query_embedding, dim=-1).cpu()
            all_query_embeddings.append(torch.tensor(query_embedding,dtype = torch.float32))
            all_query_index.append(batch["index"].cpu())

    for step, batch in tqdm(enumerate(doc_reader), total=total_doc_batches, disable=not accelerator.is_main_process):
        with torch.no_grad():
            doc_embedding = get_embed(args, model, batch["doc"],isQuery=False)
            doc_embedding = F.normalize(doc_embedding, dim=-1).cpu()
            all_candidate_embeddings.append(torch.tensor(doc_embedding,dtype = torch.float32))
            all_doc_index.append(batch["index"].cpu())

    all_query_embeddings = torch.cat(all_query_embeddings, dim=0).numpy()
    all_candidate_embeddings = torch.cat(all_candidate_embeddings, dim=0).numpy()
    all_query_index = torch.cat(all_query_index, dim=0).numpy()
    all_doc_index = torch.cat(all_doc_index, dim=0).numpy()
    with open(outpath[0], 'wb') as fout:
        pickle.dump((all_query_index, all_query_embeddings), fout)
    with open(outpath[1], 'wb') as fout:
        pickle.dump((all_doc_index, all_candidate_embeddings), fout)


def data2embeddings(query_data, doc_data, accelerator, model, output_path, args):
    query_sampler = SequentialSampler(query_data)
    doc_sampler = SequentialSampler(doc_data)
    query_reader = DataLoader(dataset=query_data, sampler=query_sampler, num_workers=args.num_workers,
                                batch_size=args.batch_size, collate_fn=query_data.Collector)
    doc_reader = DataLoader(dataset=doc_data, sampler=doc_sampler, num_workers=args.num_workers,
                                batch_size=args.batch_size, collate_fn=doc_data.Collector)
    model,query_reader,doc_reader = accelerator.prepare(model,query_reader,doc_reader)
    if args.model != 'gme' and args.model != 'bge':
        model = model.to(dtype=torch.bfloat16)
    
    gen_embeddings(model, query_reader, doc_reader, output_path, accelerator, args)


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
    sorted_pairs = sorted(zip(out_indices, out_embeddings))
    out_indices, out_embeddings = zip(*sorted_pairs)
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--mixed_precision", type=str)
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--doc_path", type=str)
    parser.add_argument("--query_path", type=str)
    parser.add_argument("--model", type=str,default=None, help='model_name: mmembed / e5-v / gme / jina / bge')
    parser.add_argument("--task_type", type=str,default=None, help='only_text')
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
    if args.model == 'mmembed':
        model = NVMMEmbedModel.from_pretrained(args.model_name_or_path,trust_remote_code=True)
        processor = None
    elif args.model == 'e5-v':
        processor = LlavaNextProcessor.from_pretrained(args.model_name_or_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    elif args.model == 'gme':
        model = GmeQwen2VL(args.model_name_or_path)
        processor = None
    elif args.model == 'bge':
        model = Visualized_BGE(model_name_bge = "BAAI/bge-base-en-v1.5", model_weight=args.model_name_or_path)
        processor = None
    elif args.model == 'jina':
        model = AutoModel.from_pretrained(args.model_name_or_path,trust_remote_code=True)
        processor = None
    query_data = WikihowInterleaved(args, model, processor, args.qrels_path, None, args.query_path, isQuery=True)
    doc_data = WikihowInterleaved(args, model, processor, args.qrels_path, args.doc_path, None, isQuery=False)
    output = [
        os.path.join(args.out_path, f'query_embedding-{accelerator.process_index}.pkl'),
        os.path.join(args.out_path, f'doc_embedding-{accelerator.process_index}.pkl')
    ]
    data2embeddings(query_data,doc_data,accelerator,model,output, args)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merge_pkl_files(args.out_path, 'query_embedding-*.pkl', os.path.join(args.out_path, f'query_embedding_{args.model}.pkl'),accelerator)
        merge_pkl_files(args.out_path, 'doc_embedding-*.pkl', os.path.join(args.out_path, f'doc_embedding_{args.model}.pkl'),accelerator)