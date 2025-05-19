import os
import json
import argparse
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.special import softmax

PROMPT_DICT_NONE = {
    "prompt_input":    "{instruction}\n{input}\n",
    "prompt_no_input": "{instruction}\n",
}

def get_perplexity_whole(tokenizer, model, text, max_length, device):
    try:
        ids = tokenizer.encode(
            text, return_tensors="pt",
            truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            out = model(ids, labels=ids)
        return torch.exp(out.loss).item()
    except:
        return float("inf")

def get_perplexity_conditional(tokenizer, model, full_text, target, max_length, device):
    try:
        ids = tokenizer.encode(
            full_text, return_tensors="pt",
            truncation=True, max_length=max_length
        ).to(device)
        decoded = tokenizer.decode(ids[0], skip_special_tokens=True)
        start = decoded.rfind(target)
        if start < 0:
            return float("inf")
        prefix_ids = tokenizer.encode(decoded[:start], return_tensors="pt").to(device)
        split = prefix_ids.shape[1]
        labels = ids.clone()
        labels[0, :split] = -100
        with torch.no_grad():
            out = model(ids, labels=labels)
        return torch.exp(out.loss).item()
    except:
        return float("inf")

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 2: compute PPL, IFD, few_shot_ifd and fs_score using DDP"
    )
    p.add_argument("--data_path",            type=str, required=True,
                   help="")
    p.add_argument("--save_path",            type=str, required=True,
                   help="")
    p.add_argument("--model_name_or_path",   type=str, default="gpt2",
                   help="")
    p.add_argument("--max_length",           type=int, default=1024,
                   help="")
    return p.parse_args()

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def main():
    args = parse_args()
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}")

    overall_start = time.time()

    t0 = time.time()
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for idx, rec in enumerate(data):
        rec.setdefault('id', idx)
    if rank == 0:
        print(f"[Load] loaded {len(data)} records in {time.time() - t0:.2f}s")

    num = len(data)
    local_idxs = list(range(rank, num, world_size))

    t1 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, output_hidden_states=False
    ).to(device)
    model = DDP(model, device_ids=[rank])
    model.eval()
    if rank == 0:
        print(f"[Model Init] model loaded and DDP-wrapped in {time.time() - t1:.2f}s")

    t2 = time.time()
    local_phase1 = {}
    for i in tqdm(local_idxs, desc=f"Rank {rank} Phase1"):
        A = data[i]
        instr, inp, outp = A['instruction'], A.get('input', "").strip(), A.get('output', "")
        prompt_A = (PROMPT_DICT_NONE["prompt_input"] if inp else PROMPT_DICT_NONE["prompt_no_input"]).format_map(A)
        ppl_A_dir  = get_perplexity_whole(tokenizer, model, outp, args.max_length, device)
        ppl_A_cond = get_perplexity_conditional(tokenizer, model, prompt_A + outp, outp, args.max_length, device)
        ifd = ppl_A_cond / ppl_A_dir if ppl_A_dir > 0 else float("inf")

        few_ifd = []
        for sim_id in A.get('top5_similar_ids', []):
            B = data[sim_id]
            instr_B, inp_B, outp_B = B['instruction'], B.get('input', "").strip(), B.get('output', "")
            prompt_B = (PROMPT_DICT_NONE["prompt_input"] if inp_B else PROMPT_DICT_NONE["prompt_no_input"]).format_map(B)
            few_ctx = prompt_A + outp + prompt_B
            ppl_B_dir  = get_perplexity_whole(tokenizer, model, outp_B, args.max_length, device)
            ppl_B_cond = get_perplexity_conditional(tokenizer, model, few_ctx + outp_B, outp_B, args.max_length, device)
            few_ifd.append(ppl_B_cond / ppl_B_dir if ppl_B_dir > 0 else float("inf"))

        local_phase1[i] = {
            "ppl_A_direct":    ppl_A_dir,
            "ppl_A_condition": ppl_A_cond,
            "ifd":             ifd,
            "few_shot_ifd":    few_ifd
        }
    if rank == 0:
        print(f"[Rank {rank}] Phase1 done in {time.time() - t2:.2f}s")

    t3 = time.time()
    gathered1 = [None] * world_size
    dist.all_gather_object(gathered1, local_phase1)
    if rank == 0:
        merged1 = {}
        for part in gathered1:
            merged1.update(part)
        for idx, rec in enumerate(data):
            if idx in merged1:
                rec.update(merged1[idx])

    dist.barrier()
    data_list = [data] if rank == 0 else [None]
    dist.broadcast_object_list(data_list, src=0)
    data = data_list[0]
    if rank == 0:
        print(f"[Gather] Phase1 merged and broadcasted in {time.time() - t3:.2f}s")

    t4 = time.time()
    local_phase2 = {}
    for i in tqdm(local_idxs, desc=f"Rank {rank} Phase2"):
        A = data[i]
        orig_ifd_list = [ data[sid]['ifd'] for sid in A.get('top5_similar_ids', []) ]
        few_ifd_list  = A.get('few_shot_ifd', [])
        fs_scores = []
        for orig_ifd, few_ifd in zip(orig_ifd_list, few_ifd_list):
            if orig_ifd > 0 and orig_ifd != float("inf"):
                fs = (orig_ifd - few_ifd) / orig_ifd
            else:
                fs = 0.0
            fs_scores.append(fs)
        local_phase2[i] = { "fs_score": fs_scores }
    if rank == 0:
        print(f"[Rank {rank}] Phase2 done in {time.time() - t4:.2f}s")

    t5 = time.time()
    gathered2 = [None] * world_size
    dist.all_gather_object(gathered2, local_phase2)
    if rank == 0:
        merged2 = {}
        for part in gathered2:
            merged2.update(part)
        for idx, rec in enumerate(data):
            if idx in merged2:
                rec.update(merged2[idx])
        with open(args.save_path, 'w', encoding='utf-8') as fw:
            json.dump(data, fw, ensure_ascii=False, indent=2)
    dist.barrier()
    if rank == 0:
        print(f"[Save] final output saved to {args.save_path} in {time.time() - t5:.2f}s")
        print(f"[Total] all steps done in {time.time() - overall_start:.2f}s")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
