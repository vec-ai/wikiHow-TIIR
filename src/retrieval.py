import sys
import csv
from tqdm import tqdm
import collections
import gzip
import pickle
import faiss
import os
import logging
import argparse
import json
import os.path as op
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pytrec_eval
import random
from msmarco_eval import quality_checks_qids, compute_metrics, load_reference
logger = logging.getLogger()

def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict


def EvalDevQuery(query_positive_id, ctx_idxs):
    prediction = {}  # [qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    total = 0
    labeled = 0
    Atotal = 0
    Alabeled = 0
    recall_1 = 0
    qids_to_ranked_candidate_passages = {}
    for query_id, top_pid in tqdm(ctx_idxs.items(), total=len(ctx_idxs)):
        seen_pid = set()
        prediction[query_id] = {}
        rank = 0

        if top_pid[0] in query_positive_id[query_id]:
            recall_1 += 1

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            # By default, all PIDs in the list of 1000 are 0. Only override those that are given
            tmp = [0] * 100
            qids_to_ranked_candidate_passages[query_id] = tmp

        for idx in top_pid:
            pred_pid = idx

            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank] = pred_pid
                Atotal += 1
                # logger.info(query_id)
                if pred_pid not in query_positive_id[query_id]:
                    Alabeled += 1
                if rank < 10:
                    total += 1
                    if pred_pid not in query_positive_id[query_id]:
                        labeled += 1
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(query_positive_id), {'ndcg_cut', 'recall'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))

    qids_to_relevant_passageids = {}
    for qid in tqdm(query_positive_id):
        if qid in qids_to_relevant_passageids:
            pass
        else:
            qids_to_relevant_passageids[qid] = []
            for pid in query_positive_id[qid]:
                qids_to_relevant_passageids[qid].append(pid)

    mrr_5 = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 5)
    mrr_10 = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 10)
    mrr_20 = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 20)

    ndcg_5 = 0
    ndcg_10 = 0
    ndcg_20 = 0
    recall_5 = 0
    recall_100 = 0
    recall_20 = 0

    for k in tqdm(result.keys()):
        eval_query_cnt += 1
        recall_5 += result[k]["recall_5"]
        recall_100 += result[k]["recall_100"]
        recall_20 += result[k]["recall_20"]
        ndcg_5 += result[k]["ndcg_cut_5"]
        ndcg_10 += result[k]["ndcg_cut_10"]
        ndcg_20 += result[k]["ndcg_cut_20"]

    recall_1 = recall_1 / eval_query_cnt
    recall_5 = recall_5 / eval_query_cnt
    recall_100 = recall_100 / eval_query_cnt
    recall_20 = recall_20 / eval_query_cnt
    ndcg_5 = ndcg_5 / eval_query_cnt
    ndcg_10 = ndcg_10 / eval_query_cnt
    ndcg_20 = ndcg_20 / eval_query_cnt

    return recall_1, recall_5, recall_20, recall_100, mrr_10, mrr_20, ndcg_10, ndcg_20, qids_to_ranked_candidate_passages, mrr_5, ndcg_5,

if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--query_embed_path")
    parser.add_argument("--doc_embed_path")
    parser.add_argument("--out_path", default='./')

    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--topN", type=int, default=100)
    parser.add_argument("--query_num", type=int, default=None)
    parser.add_argument("--out_topk_path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)


    args = parser.parse_args()


    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'evaluation_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)

    logger.info(args)
    all_idx = []
    all_embeds = []
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(args.dim)
    if args.doc_embed_path:
        logger.info("load data from {}".format(args.doc_embed_path))
        with open(args.doc_embed_path, 'rb') as fin:
            doc_idx, doc_embeds = pickle.load(fin)
            cpu_index.add(np.array(doc_embeds, dtype=np.float32))
            del doc_embeds
            all_idx.extend(doc_idx)

    with open(args.query_embed_path, 'rb') as fin:
        logger.info("load data from {}".format(args.query_embed_path))
        query_idx, query_embeds = pickle.load(fin)

    logger.info("Retrieve candidates!")
    query_embeds = np.array(query_embeds, dtype=np.float32)
    if args.query_num is not None:
        query_idx, query_embeds = query_idx[:args.query_num], query_embeds[:args.query_num]
    D, I = cpu_index.search(query_embeds, args.topN)
    ctx_idxs = {}
    assert len(query_idx) == len(I)
    for step, qid in enumerate(query_idx):
        ctx_idxs[qid] = []
        for idx in I[step]:
            ctx_idxs[qid].append(all_idx[idx])
    query_positive_id = {}
    for i in range(len(query_embeds)):
        query_positive_id[query_idx[i]] = {}
        query_positive_id[query_idx[i]][all_idx[i]] = 1


    result = EvalDevQuery(query_positive_id, ctx_idxs)
    recall_1, recall_5, recall_20, recall_100, mrr_10, mrr_20, ndcg_10, ndcg_20, qids_to_ranked_candidate_passages, mrr_5, ndcg_5, = result
    logger.info("recall@1:{:.2f}".format(round(recall_1 * 100, 2)))
    logger.info("recall@5:{:.2f}".format(round(recall_5 * 100, 2)))
    logger.info("recall@20:{:.2f}".format(round(recall_20 * 100, 2)))
    logger.info("recall@100:{:.2f}".format(round(recall_100 * 100, 2)))

    logger.info("mrr@5:{:.2f}".format(round(mrr_5 * 100, 2)))
    logger.info("ndcg@5:{:.2f}".format(round(ndcg_5 * 100, 2)))
    logger.info("mrr@10:{:.2f}".format(round(mrr_10 * 100, 2)))
    logger.info("ndcg@10:{:.2f}".format(round(ndcg_10 * 100, 2)))
    logger.info("mrr@20:{:.2f}".format(round(mrr_20 * 100, 2)))
    logger.info("ndcg@20:{:.2f}".format(round(ndcg_20 * 100, 2)))

    with open(f'{args.out_path}/results_{args.model}.txt', 'w') as f:
        results = [
            recall_1,
            recall_5,
            recall_20,
            recall_100,
            mrr_5,
            mrr_10,
            mrr_20,
            ndcg_5,
            ndcg_10,
            ndcg_20
        ]
        result_strings = [str(round(result * 100, 2)) for result in results]
        result_line = '|'.join(result_strings)
        f.write(result_line)

    if args.out_topk_path is not None:
        with open(os.path.join(args.out_topk_path, 'topk_passages.pkl'), 'wb') as fout:
            pickle.dump(qids_to_ranked_candidate_passages, fout)
        top1_passages_idx = [k for k, v in qids_to_ranked_candidate_passages.items() if k==v[0]]
        with open(os.path.join(args.out_topk_path, 'top1_passages_idx.pkl'), 'wb') as fout:
            pickle.dump(top1_passages_idx, fout)
        
