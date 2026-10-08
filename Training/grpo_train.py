"""GRPO training for a Mastodon search-query agent.

The policy reads a post's text and writes a short search query. The reward runs
scripts/first-search-result.js with that query and gives 1.0 if the first result's URL matches
the post's URL, otherwise 0.0.

Usage:
    python scripts/grpo_train.py
"""
import json
import os
import random
import subprocess
import time

import torch
from functools import lru_cache

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEARCH_SCRIPT = os.path.join(REPO_ROOT, "scripts", "first-search-result.js")

INSTRUCTION = (
    "Write a short search query (a few keywords, no quotes, no explanation) that would find "
    "the following Mastodon post with a search engine.\n\nPost:\n"
)

CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "data": os.path.join(REPO_ROOT, "data", "dataset.json"),
    "field": "paraphrased_root_post_text",  # which text field the agent sees as the phrase to search for
    "instance": "mastodon.social",
    "output": os.path.join(REPO_ROOT, "outputs", "grpo_search"),
    "limit": 0,
    "epochs": 1.0,
    "lr": 5e-6,
    "num_generations": 5,
    "batch_size": 5,
    "no_save": False,
    "no_lora": False,
}


def normalize_url(url):
    return (url or "").strip().rstrip("/")


def load_records(path, field, limit):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    records = []
    for row in rows:
        text = (row.get(field) or "").strip()
        url = normalize_url(row.get("post_url"))
        if not text or not url:
            continue
        records.append({
            "prompt": [{"role": "user", "content": INSTRUCTION + text}],
            "post_url": url,
        })
    if limit:
        records = records[:limit]
    return Dataset.from_list(records)


def extract_query(completion):
    text = completion[0]["content"] if isinstance(completion, list) else completion
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line.strip().strip('"').strip("'").strip()


@lru_cache(maxsize=None)
def search_first_url(query, instance):
    proc = subprocess.run(
        ["node", SEARCH_SCRIPT, query, instance],
        capture_output=True, text=True, timeout=180, cwd=REPO_ROOT,
    )
    time.sleep(random.uniform(1.0, 2.0))
    if proc.returncode != 0:
        return None
    lines = proc.stdout.strip().splitlines()
    result = lines[-1].strip() if lines else ""
    return None if result in ("", "None") else normalize_url(result)


def make_reward_fn(instance):
    def reward_fn(prompts, completions, post_url, **kwargs):
        rewards = []
        for completion, target in zip(completions, post_url):
            query = extract_query(completion)
            if not query:
                rewards.append(0.0)
                continue
            found = search_first_url(query, instance)
            rewards.append(1.0 if found == normalize_url(target) else 0.0)
        return rewards

    return reward_fn


def main():
    cfg = CONFIG

    dataset = load_records(cfg["data"], cfg["field"], cfg["limit"])
    print(f"training records: {len(dataset)}")

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
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=make_reward_fn(cfg["instance"]),
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    if not cfg["no_save"]:
        trainer.save_model(cfg["output"])
        print(f"saved to {cfg['output']}")


if __name__ == "__main__":
    main()
