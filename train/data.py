import json
import os
import torch
import random
from PIL import Image
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import torch
from PIL import ImageFile
import pyarrow as pa
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True
import datasets
from tqdm import tqdm
import random


class WikihowInterleavedData:
    def __init__(self, instance):
        self.vl_data_query = instance['vl_data_query'] if 'vl_data_query' in instance else None
        self.vl_data_doc = instance['vl_data_doc'] if 'vl_data_doc' in instance else None
        self.image_query = instance['image_query'] if 'image_query' in instance else None
        self.image_doc = instance['image_doc'] if 'image_doc' in instance else None

        self.vl_data_negquery = instance['vl_data_negquery'] if 'vl_data_negquery' in instance else None
        self.vl_data_negdoc = instance['vl_data_negdoc'] if 'vl_data_negdoc' in instance else None
        self.image_negquery = instance['image_negquery'] if 'image_negquery' in instance else None
        self.image_negdoc = instance['image_negdoc'] if 'image_negdoc' in instance else None


class WikihowInterleaved(Dataset):
    def __init__(self, args, preprocess_fn, vl_chat_processor, qrels_path, doc_path, query_path, doc_image_root, query_image_root, shuffle):
        self.shuffle = shuffle
        self.preprocess_fn = preprocess_fn
        self.vl_chat_processor=vl_chat_processor

        self.doc_image_root = doc_image_root
        self.query_image_root = query_image_root
        self.qrels_path = qrels_path
        self.doc_path = doc_path
        self.query_path = query_path

        self.qrels = []
        self.docs = {}
        self.queries = {}
        self.negids = {}

        self.init()


    def init(self):
        ids = []
        with open(self.qrels_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.qrels.append((data['did'],data['qid']))
                self.negids[data['did']] = data['neg']
                ids.append(data['did']) 
        if self.shuffle:
            random.shuffle(self.qrels)
        with open(self.doc_path,'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.docs[data['id']] = {'data':data['data']}
                self.docs[data['id']]['vl_data'] = self.create_deepseek_conversations(data['data'])

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
                images.append(d)
                if text is None:
                    text = "<image_placeholder> "
                else:
                    text += "\n <image_placeholder>"
            else:
                if text is None:
                    text = d
                else:
                    text += f"\n {d}"
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
        processed_batch = {}
        for qid, example in enumerate(batch):
            vl_data_query.append(example.vl_data_query)
            vl_data_doc.append(example.vl_data_doc)  
            vl_data_doc.append(example.vl_data_negdoc)  
            image_query.append(example.image_query)
            image_doc.append(example.image_doc)
            image_doc.append(example.image_negdoc)

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

        return processed_batch


    def __getitem__(self, index):
        data_id = self.qrels[index][0]
        neg_id = self.negids[data_id]
        query,qimages = self.queries[data_id]['vl_data']
        doc,dimages = self.docs[data_id]['vl_data']
        
        for neg in neg_id:
            if neg not in self.queries:
                neg_id.remove(neg)
        if len(neg_id) == 0:
            idx = random.randint(0, len(self.qrels)-1)
            while(self.qrels[idx][0] not in self.queries):
                idx = random.randint(0, len(self.qrels)-1)
            neg_id.append(self.qrels[idx][0])
        neg_id = random.choice(neg_id)
        neg_doc,neg_dimages = self.docs[neg_id]['vl_data']

        instance = {}
        instance['vl_data_query'] = query
        instance['vl_data_doc'] = doc
        instance['image_query'] = self.encode_imgs(qimages)
        instance['image_doc'] = self.encode_imgs(dimages)

        instance['vl_data_negdoc'] = neg_doc
        instance['image_negdoc'] = self.encode_imgs(neg_dimages)
        return WikihowInterleavedData(instance)





