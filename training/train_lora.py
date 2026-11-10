"""LoRA 微调：让 Qwen2.5-1.5B-Instruct 稳定输出符合契约的对话树 JSON。

用法（项目根目录 npc_research/ 下执行）：
    python training/train_lora.py --epochs 2 --lr 2e-4
输出：training/output/lora/（适配器权重），training/train_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]


class PairDataset(Dataset):
    def __init__(self, path: str, tok, max_len: int):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        ex = self.rows[i]
        prompt = self.tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = self.tok(ex["completion"] + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + c_ids
        if len(ids) > self.max_len:
            # 保留完整答案，从左侧裁剪提示
            keep_c = min(len(c_ids), self.max_len - 32)
            keep_p = self.max_len - keep_c
            ids = p_ids[-keep_p:] + c_ids[-keep_c:]
            labels = [-100] * keep_p + c_ids[-keep_c:]
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}


def collate(batch: list[dict], pad_id: int) -> dict:
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--data", default=str(ROOT / "training" / "data"))
    ap.add_argument("--out", default=str(ROOT / "training" / "output" / "lora"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=1408)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[train] device={device}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if device == "cuda":
        model = model.to("cuda")

    ds = PairDataset(str(Path(args.data) / "train.jsonl"), tok, args.max_len)
    dl = DataLoader(
        ds, batch_size=args.micro_batch, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id),
    )
    steps_per_epoch = math.ceil(len(dl) / args.accum)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, total_steps), pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    log_path = ROOT / "training" / "train_log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow(["step", "loss", "lr", "elapsed_sec"])

    model.train()
    step, t0, running, n_micro = 0, time.time(), 0.0, 0
    done = False
    for epoch in range(math.ceil(args.epochs)):
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            ids, lab, att = batch["input_ids"], batch["labels"], batch["attention_mask"]
            # 只对答案 token 计算 logits（词表 15 万，全序列 logits 会吃满 8GB 显存）
            k = int((lab != -100).sum(dim=1).max().item()) + 1
            k = max(2, min(k, ids.shape[1]))
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out = model(input_ids=ids, attention_mask=att, logits_to_keep=k)
                logits = out.logits[:, :-1, :].float()
                targets = lab[:, 1:][:, -(k - 1):]
                loss = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), targets, ignore_index=-100
                ) / args.accum
            scaler.scale(loss).backward()
            running += loss.item()
            n_micro += 1
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 5 == 0 or step == 1:
                    el = time.time() - t0
                    avg_loss = running / n_micro
                    print(f"[train] step {step}/{total_steps} loss={avg_loss:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} {el:.0f}s", flush=True)
                    writer.writerow([step, round(avg_loss, 5), f"{sched.get_last_lr()[0]:.3e}", round(el, 1)])
                    log_f.flush()
                    running, n_micro = 0.0, 0
                if args.save_every and step % args.save_every == 0:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    ck = Path(args.out) / f"step_{step}"
                    ck.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ck))
                    print(f"[train] checkpoint -> {ck}", flush=True)
                if step >= total_steps:
                    done = True
                    break
        if done:
            break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    log_f.close()
    print(f"[train] saved adapter -> {out}, steps={step}, total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
