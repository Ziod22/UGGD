# UGGD

Code for a two-part pipeline on Mastodon:

1. **Post retrieval**: a language model is trained to write a short search query that finds a given
   Mastodon post from a reworded description of it.
2. **Rumour classification**: the found post and its reply thread are turned into a propagation tree
   and classified by a BiGCN graph model.

```
reworded post ──► query model ──► Mastodon search ──► post URL
                                                         │
             BiGCN prediction ◄── propagation tree ◄── fetch thread
             (load_bigcn.py)      (preprocess.py)      (fetch.js)
```

## Repository layout

```
UGGD/
├── README.md
├── pipeline_demo_page_template.html
├── Training/
│   ├── grpo_train.py
│   ├── grpo_top10_paraphrase.py
│   ├── first-search-result.js
│   └── search_worker.js
└── Model_pipeline/
    ├── fetch.js
    ├── preprocess.py
    └── load_bigcn.py
```

## Training/

Scripts that train the query model with GRPO (TRL) and the Node.js helpers that run Mastodon searches
for the reward.

| File | What it does |
|---|---|
| `grpo_train.py` | First GRPO training script. The model (`Qwen/Qwen2.5-0.5B-Instruct`, LoRA r=16) reads a post's text (field `paraphrased_root_post_text` of `data/dataset.json`) and writes a search query. Each query is searched with `first-search-result.js`; reward 1.0 if the **first** result's URL is the post's URL, else 0.0. 5 tries per prompt, lr 5e-6, 1 epoch, max 32 new tokens. Saves to `outputs/grpo_search/`. |
| `grpo_top10_paraphrase.py` | Second GRPO training script with a softer reward. The model (`Qwen/Qwen3.5-4B`, LoRA r=16) writes a query; `search_worker.js` returns the **top 10** results; each result text is embedded with `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` and compared with the post's text. Reward 1.0 if any result reaches similarity 0.7, else 0.0. Before training it checks which posts are searchable at all (the original text finds its own URL) and keeps only those (`searchable_cache.json`, `searchable_check.jsonl`). Every try is logged to `audit.jsonl`. Data `data/expanded/dataset_expanded.json`, output `outputs/top10_paraphrase/`. |
| `first-search-result.js` | `node first-search-result.js <query> [instance]`: prints the URL of the first `/api/v2/search` status result (or `None`). Uses Playwright and a saved logged-in browser session (`tmp/auth/mastodon_<instance>.json`) to get an access token. Starts one browser per call, so it is slow; used by `grpo_train.py`. |
| `search_worker.js` | Long-running version of the search used by `grpo_top10_paraphrase.py`. Reads one JSON request per line on stdin (`{"id", "query"}`) and writes one reply per line on stdout (`{"id", "results": [{"url", "text"}]}` or `{"id", "error"}`). One browser and token are reused for all queries; HTML is stripped from result text; on HTTP 401 the token is refreshed. Searches run one at a time with a random 1–2 s pause (rate limit). Env: `MASTODON_INSTANCE` (default `mastodon.social`), `SEARCH_TOP_N` (default 10). |

## Model_pipeline/

Inference side: fetch the found post's thread, build the propagation tree, classify it.

| File | What it does |
|---|---|
| `fetch.js` | `node fetch.js <post-url> [--output-dir <dir>]`: fetches one public post from its home instance with the public API (no login): the post (`/api/v1/statuses/<id>`; for a boost, the original), all replies (`/context` descendants: id, author, text, time, `inReplyToId`) and everyone who favourited it (`/favourited_by`, paginated, max 25 pages). Also keeps counts, media, tags, language and visibility. Saves `<instance>_<postId>.json`, the input of `preprocess.py`. Needs Node 18+ (built-in `fetch`). |
| `preprocess.py` | Turns `fetch.js` files into BiGCN's `.npz` graph format. Per thread: the root post is node 0 and each reply a further node; edges go parent → child from `inReplyToId` (replies whose parent was not fetched get no edge); each node's text becomes a bag-of-words vector over the model's vocabulary (`--vocab vocab.json`, 5000 words, 1-based indices). Saves `<rootId>.npz` with `x` (nodes × vocab), `root`, `edgeindex` (2 × edges), `rootindex`, and `y` only if `--labels` gives one; unlabelled threads are listed in `labels_template.txt`. `--min-replies N` skips small threads. |
| `load_bigcn.py` | Rebuilds the BiGCN model (Bian et al., AAAI 2020, [TianBian95/BiGCN](https://github.com/TianBian95/BiGCN)): a top-down GCN over the reply edges and a bottom-up GCN over the reversed edges, each with root-feature enhancement, concatenated and classified. Loads a trained checkpoint and one `.npz` graph and prints the predicted class and class probabilities. `--dataset Twitter15`/`Twitter16`: non-rumor, false, true, unverified; `Weibo`: false, true. |

Example:

```
node Model_pipeline/fetch.js https://mastodon.social/@user/123456789 --output-dir output
python Model_pipeline/preprocess.py --input "output/mastodon.social_*.json" --vocab vocab.json --output-dir output
python Model_pipeline/load_bigcn.py --checkpoint <bigcn_checkpoint.pt> --input output/123456789.npz --dataset Twitter15
```

## pipeline_demo_page_template.html

Self-contained web page (open it in a browser, no build step) that shows the whole pipeline as a step
graph: user input → generated queries → search results → candidate pool → chosen post → fetched thread →
propagation tree (drawn as a tree) → BiGCN prediction, with details for each step. It ships with no
example data: it can load a pipeline `trace.json` (link at the bottom of the left panel), and its
"Live run on the HPC" panel talks to a backend at `http://localhost:8765` that runs the pipeline on a GPU
cluster. That backend is **not** included (it uses an SSH key and connects to the cluster server), so
without it the live panel shows the server as offline. The page was generated with claude-opus-5.5
(noted in a comment at the top of the file).

## Requirements and notes

- **Python**: `torch`, `transformers`, `trl`, `peft`, `datasets`, `sentence-transformers` (training);
  `torch`, `torch_geometric`, `torch_scatter`, `numpy` (BiGCN).
- **Node.js**: 18+; `@playwright/test` with Chromium for the two search scripts.
- The search scripts need a saved logged-in Mastodon session in `tmp/auth/mastodon_<instance>.json`;
  the login helper and the session file are not part of this repository.
- The training scripts expect the project layout they were written in: helpers in `scripts/`
  (here they are in `Training/`), datasets in `data/`, outputs in `outputs/`. The datasets, trained
  weights, the BiGCN checkpoint and `vocab.json` are not included.
- Search results come from the live mastodon.social index (all query words must match; newest posts
  first), so results change over time.
