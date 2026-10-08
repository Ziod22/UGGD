"""GRPO training with a top-10 paraphrase-match reward.

The policy reads a post's paraphrased text and writes a short search query. The reward runs the
search worker (scripts/search_worker.js) for the query, takes the top 10 result texts, embeds
them with a paraphrase-detection model, and compares each one against the dataset's paraphrased
text for the post. The try is a success (reward 1.0) if any of the 10 results reaches the
similarity threshold; otherwise 0.0.

Usage:
    python scripts/grpo_top10_paraphrase.py
"""
import json
import os
import subprocess
import sys

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from trl import GRPOConfig, GRPOTrainer

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WORKER = os.path.join(REPO_ROOT, "scripts", "search_worker.js")

INSTRUCTION = (
    "Write a short search query (a few keywords, no quotes, no explanation) that would find "
    "the following Mastodon post with a search engine.\n\nPost:\n"
)

CONFIG = {
    "model": "Qwen/Qwen3.5-4B",
    "data": os.path.join(REPO_ROOT, "data", "expanded", "dataset_expanded.json"),
    "embed_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "threshold": 0.7,
    "top_n": 10,
    "instance": "mastodon.social",
    "output": os.path.join(REPO_ROOT, "outputs", "top10_paraphrase"),
    "limit": 0,
    "epochs": 1.0,
    "lr": 5e-6,
    "num_generations": 5, 
    "batch_size": 5,
    "no_save": False,
    "no_lora": False,
    "check_searchable": True,
}


class SearchWorker:
    def __init__(self, instance, top_n, log_path):
        self.instance = instance
        self.top_n = top_n
        self.log_path = log_path
        self.next_id = 0
        self.proc = None
        self.start()

    def start(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = subprocess.Popen(
            ["node", WORKER],
            cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(self.log_path, "a"), text=True, bufsize=1,
            env={**os.environ, "MASTODON_INSTANCE": self.instance, "SEARCH_TOP_N": str(self.top_n)},
        )

    def query(self, text):
        self.next_id += 1
        try:
            self.proc.stdin.write(json.dumps({"id": self.next_id, "query": text}) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except (BrokenPipeError, OSError):
            line = ""
        if not line:
            self.start()
            return {"error": "worker restarted"}
        return json.loads(line)


def extract_query(completion):
    text = completion[0]["content"] if isinstance(completion, list) else completion
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line.strip().strip('"').strip("'").strip()


def normalize_url(url):
    return (url or "").strip().rstrip("/")


def searchable_urls(rows, worker, cache_path, log):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for row in rows:
        url = normalize_url(row.get("post_url"))
        if not url or url in cache:
            continue
        query = " ".join((row.get("original_root_post_text") or "").split()[:10])
        reply = worker.query(query)
        if "error" in reply:
            cache[url] = {"searchable": False, "reason": reply["error"]}
        else:
            found = any(normalize_url(r.get("url")) == url for r in reply.get("results", []))
            cache[url] = {"searchable": found, "reason": "found" if found else "not in top results"}
        json.dump(cache, open(cache_path, "w"), ensure_ascii=False)
        log.write(json.dumps({"url": url, **cache[url]}, ensure_ascii=False) + "\n")
        log.flush()
    return {u for u, v in cache.items() if v["searchable"]}


def load_records(path, limit, keep_urls=None):
    rows = json.load(open(path, encoding="utf-8"))
    records = []
    for row in rows:
        if keep_urls is not None and normalize_url(row.get("post_url")) not in keep_urls:
            continue
        text = (row.get("root_post_text") or row.get("paraphrased_root_post_text") or "").strip()
        if not text:
            continue
        records.append({
            "prompt": [{"role": "user", "content": INSTRUCTION + text}],
            "target_text": text,
            "post_url": (row.get("post_url") or "").rstrip("/"),
        })
    if limit:
        records = records[:limit]
    return Dataset.from_list(records)


def make_reward(worker, embedder, threshold, cache, audit_path):
    audit = open(audit_path, "a", encoding="utf-8")

    def reward_fn(prompts, completions, target_text, post_url, **kwargs):
        rewards = []
        for completion, target, url in zip(completions, target_text, post_url):
            query = extract_query(completion)
            if not query:
                rewards.append(0.0)
                audit.write(json.dumps({"url": url, "query": "", "max_sim": None, "reward": 0.0}) + "\n")
                continue
            if query not in cache:
                reply = worker.query(query)
                cache[query] = (
                    [r["text"] for r in reply.get("results", []) if r.get("text")]
                    if "error" not in reply else []
                )
            texts = cache[query]
            if not texts:
                reward, max_sim = 0.0, None
            else:
                emb = embedder.encode(texts + [target], normalize_embeddings=True, convert_to_numpy=True)
                max_sim = float((emb[:-1] @ emb[-1]).max())
                reward = 1.0 if max_sim >= threshold else 0.0
            rewards.append(reward)
            audit.write(json.dumps({"url": url, "query": query, "n_results": len(texts),
                                    "max_sim": max_sim, "reward": reward}, ensure_ascii=False) + "\n")
        audit.flush()
        return rewards

    return reward_fn


def main():
    cfg = CONFIG

    os.makedirs(cfg["output"], exist_ok=True)
    worker = SearchWorker(cfg["instance"], cfg["top_n"], os.path.join(cfg["output"], "worker.log"))
    keep_urls = None
    if cfg["check_searchable"]:
        rows = json.load(open(cfg["data"], encoding="utf-8"))
        with open(os.path.join(cfg["output"], "searchable_check.jsonl"), "a", encoding="utf-8") as log:
            keep_urls = searchable_urls(rows, worker, os.path.join(cfg["output"], "searchable_cache.json"), log)
        print(f"searchable posts: {len(keep_urls)}", flush=True)
    dataset = load_records(cfg["data"], cfg["limit"], keep_urls)
    print(f"training records: {len(dataset)}", flush=True)
    embedder = SentenceTransformer(cfg["embed_model"], device="cuda")
    reward_fn = make_reward(worker, embedder, cfg["threshold"], {}, os.path.join(cfg["output"], "audit.jsonl"))

    config = GRPOConfig(
        output_dir=cfg["output"],
        model_init_kwargs={"dtype": torch.bfloat16},
        gradient_checkpointing=True,
        chat_template_kwargs={"enable_thinking": False},
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        num_generations=cfg["num_generations"],
        max_completion_length=32,
        logging_steps=1,
        save_strategy="no" if cfg["no_save"] else "epoch",
        bf16=True,
        report_to="none",
    )

    peft_config = None
    if not cfg["no_lora"]:
        from peft import LoraConfig
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    if not cfg["no_save"]:
        trainer.save_model(cfg["output"])
        print(f"saved to {cfg['output']}", flush=True)
    worker.proc.kill()


if __name__ == "__main__":
    sys.exit(main())
