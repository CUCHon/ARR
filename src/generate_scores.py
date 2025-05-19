#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time

import faiss
import numpy as np
import torch
import torch.distributed as dist
from scipy.special import softmax
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1: compute top-K similar IDs, quality_score, complexity_score using DDP."
    )
    parser.add_argument("--input_path",       type=str, required=True,
                        help="Path to input JSON file (list of dicts with instruction,input,output)")
    parser.add_argument("--output_path",      type=str, required=True,
                        help="Path to write output JSON (only rank 0 writes)")
    parser.add_argument("--sbert_model",      type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--cross_model",      type=str, default="cross-encoder/stsb-distilroberta-base")
    parser.add_argument("--quality_model",    type=str, default="hkust-nlp/deita-quality-scorer")
    parser.add_argument("--complexity_model", type=str, default="hkust-nlp/deita-complexity-scorer")
    parser.add_argument("--max_len",          type=int, default=512,
                        help="Truncate inputs to this length")
    parser.add_argument("--top_k",            type=int, default=5,
                        help="How many similar samples to keep")
    parser.add_argument("--batch_size",       type=int, default=16,
                        help="Batch size for quality/complexity scoring")
    return parser.parse_args()

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for idx, rec in enumerate(data):
        rec['id'] = idx
    return data

def save_data(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def compute_similarities(data, sbert_model_name, cross_model_name, top_k):
    inst = [rec['instruction'] for rec in data]
    st = SentenceTransformer(sbert_model_name)
    emb = st.encode(inst, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    sims, idxs = index.search(emb, top_k+1)

    ce = CrossEncoder(cross_model_name)
    for i, rec in enumerate(tqdm(data, desc="CrossEncoder rerank")):
        cands = [int(x) for x in idxs[i] if x != i][:top_k]
        pairs = [[rec['instruction'], data[j]['instruction']] for j in cands]
        scores = ce.predict(pairs)
        order = np.argsort(-scores)
        rec[f"top{top_k}_similar_ids"] = [cands[o] for o in order[:top_k]]

    del st, emb, index, ce
    torch.cuda.empty_cache()

def batch_infer(model, tokenizer, prompts, max_len, device, max_new_tokens=5):
    gen = model.module if hasattr(model, "module") else model
    enc = tokenizer(prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len).to(device)
    out = gen.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True
    )
    return out.scores[0]  # (batch_size, vocab_size)

def score_batch_logits(logits, id2score):
    lp = logits.detach().cpu().numpy()
    idxs = list(id2score.keys())
    weights = np.array(list(id2score.values()), dtype=float)
    sub = lp[:, idxs]           # (B,6)
    probs = softmax(sub, axis=1)
    return (probs * weights[None, :]).sum(axis=1)  # (B,)

def main():
    args = parse_args()
    rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    overall_start = time.time()
    data = load_data(args.input_path)
    if rank == 0:
        print(f"[Load] {len(data)} records loaded in {time.time() - overall_start:.2f}s")

    t1 = time.time()
    compute_similarities(data, args.sbert_model, args.cross_model, args.top_k)
    if rank == 0:
        print(f"[Similarities] done in {time.time() - t1:.2f}s")

    t2 = time.time()
    q_tok = AutoTokenizer.from_pretrained(args.quality_model)
    q_mod = AutoModelForCausalLM.from_pretrained(args.quality_model).to(device)
    if world_size > 1:
        q_mod = torch.nn.parallel.DistributedDataParallel(q_mod, device_ids=[rank])
    q_mod.eval()

    q_prompts = [
        rec['instruction'] + ("\n" + rec['input'] if rec.get('input') else "") +
        "\n#Response#:\n" + rec.get('output', "") +
        "\n##Quality:" for rec in data
    ]
    local_idxs = list(range(rank, len(data), world_size))
    local_q = {}
    id2s = {29896:1,29906:2,29941:3,29946:4,29945:5,29953:6}

    for i in tqdm(range(0, len(local_idxs), args.batch_size), desc="Quality scoring"):
        batch_idxs = local_idxs[i:i+args.batch_size]
        batch_prompts = [q_prompts[j] for j in batch_idxs]
        logits = batch_infer(q_mod, q_tok, batch_prompts, args.max_len, device)
        scores = score_batch_logits(logits, id2s)
        for k, idx in enumerate(batch_idxs):
            local_q[idx] = float(scores[k])

    gathered_q = [None] * world_size
    dist.all_gather_object(gathered_q, local_q)
    if rank == 0:
        # merge
        quality_map = {}
        for d in gathered_q:
            quality_map.update(d)
        for rec in data:
            rec['quality_score'] = quality_map[rec['id']]
        print(f"[Quality] done in {time.time() - t2:.2f}s")
    dist.barrier()
    del q_mod; torch.cuda.empty_cache()

    t3 = time.time()
    c_tok = AutoTokenizer.from_pretrained(args.complexity_model)
    c_mod = AutoModelForCausalLM.from_pretrained(args.complexity_model).to(device)
    if world_size > 1:
        c_mod = torch.nn.parallel.DistributedDataParallel(c_mod, device_ids=[rank])
    c_mod.eval()

    c_prompts = [
        rec['instruction'] + ("\n" + rec['input'] if rec.get('input') else "") +
        "\n##Complexity:" for rec in data
    ]
    local_c = {}
    for i in tqdm(range(0, len(local_idxs), args.batch_size), desc="Complexity scoring"):
        batch_idxs = local_idxs[i:i+args.batch_size]
        batch_prompts = [c_prompts[j] for j in batch_idxs]
        logits = batch_infer(c_mod, c_tok, batch_prompts, args.max_len, device)
        scores = score_batch_logits(logits, id2s)
        for k, idx in enumerate(batch_idxs):
            local_c[idx] = float(scores[k])

    gathered_c = [None] * world_size
    dist.all_gather_object(gathered_c, local_c)
    if rank == 0:
        complexity_map = {}
        for d in gathered_c:
            complexity_map.update(d)
        for rec in data:
            rec['complexity_score'] = complexity_map[rec['id']]
        print(f"[Complexity] done in {time.time() - t3:.2f}s")
    dist.barrier()
    del c_mod; torch.cuda.empty_cache()

    if rank == 0:
        save_data(args.output_path, data)
        print(f"[Save] {time.time() - overall_start:.2f}s → {args.output_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
