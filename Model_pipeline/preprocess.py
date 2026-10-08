"""Preprocess Mastodon post and reply-thread JSON files (from fetch.js) into BiGCN's .npz propagation-tree graph format.

Input: output files from fetch.js

Pipeline, per thread:
  1. Assign every post (root + replies) a 0-based node index.
  2. Build parent->child edges from each reply's inReplyToId.
  3. Encode each node's text into the trained model's vocabulary (bag-of-words)
  4. Save an .npz with the same keys: x, root, edgeindex, rootindex, y.

Usage:
    python preprocess.py \
        --input "mastodon.social_*.json" \
        --vocab vocab.json \
        --output-dir output
"""
import argparse
import glob
import json
import os
import numpy as np


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def is_thread_file(obj):
    return isinstance(obj, dict) and 'id' in obj and 'text' in obj and 'replies' in obj


def collect_nodes(thread):
    nodes = [(thread['id'], None, thread.get('text') or '')]
    for reply in thread['replies']:
        nodes.append((reply['id'], reply.get('inReplyToId'), reply.get('text') or ''))
    return nodes


def build_graph(thread, word_to_index, vocab_size):
    nodes = collect_nodes(thread)
    id_to_index = {post_id: i for i, (post_id, _parent, _text) in enumerate(nodes)}
    row, col = [], []
    skipped = 0
    for i, (_post_id, parent_id, _text) in enumerate(nodes):
        if parent_id is None:
            continue  # root has no parent edge
        if parent_id not in id_to_index:
            skipped += 1  # parent wasn't fetched (e.g. a reply to a reply outside the thread)
            continue
        row.append(id_to_index[parent_id])
        col.append(i)
    x = np.zeros((len(nodes), vocab_size), dtype=np.float64)
    for i, (_post_id, _parent, text) in enumerate(nodes):
        for token in text.lower().split():
            idx = word_to_index.get(token)
            if idx is not None:
                x[i, idx - 1] += 1  # word_to_index is 1-based
    root_feat = x[0:1, :].copy()
    edgeindex = [row, col]
    return x, root_feat, edgeindex, len(nodes), skipped


def load_vocab(vocab_path):
    with open(vocab_path, encoding='utf-8') as f:
        vocab_data = json.load(f)
    word_to_index = vocab_data['word_to_index']
    return word_to_index, len(word_to_index)


def preprocess(paths, word_to_index, vocab_size, output_dir, labels=None, min_replies=0):
    labels = labels or {}
    threads = []
    below_threshold = 0
    for path in paths:
        obj = load_json(path)
        if not is_thread_file(obj):
            print('File illegal): {}'.format(path))
            continue
        if len(obj['replies']) < min_replies:
            below_threshold += 1
            continue
        threads.append(obj)
    if not threads:
        raise SystemExit('thread not meeting {} replies requirement'.format(min_replies))

    os.makedirs(output_dir, exist_ok=True)
    written = []
    unlabeled_ids = []
    for thread in threads:
        root_id = thread['id']
        x, root_feat, edgeindex, num_nodes, skipped = build_graph(thread, word_to_index, vocab_size)
        save_kwargs = dict(x=x, root=root_feat, edgeindex=np.array(edgeindex), rootindex=np.array(0))
        if root_id in labels:
            label = labels[root_id]
            save_kwargs['y'] = np.array(int(label)) if label.isdigit() else np.array(label)
        else:
            unlabeled_ids.append(root_id)
        out_path = os.path.join(output_dir, '{}.npz'.format(root_id))
        np.savez(out_path, **save_kwargs)
        written.append((root_id, out_path))
        print('  Wrote {} ({} nodes, {} edges)'.format(out_path, num_nodes, len(edgeindex[0])))

    if unlabeled_ids:
        template_path = os.path.join(output_dir, 'labels_template.txt')
        with open(template_path, 'w', encoding='utf-8') as f:
            for post_id in unlabeled_ids:
                f.write('{}\tunknown\n'.format(post_id))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--labels', default=None)
    parser.add_argument('--min-replies', type=int, default=0,)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input))
    if not paths:
        raise SystemExit('No files matched {}'.format(args.input))

    word_to_index, vocab_size = load_vocab(args.vocab)
    print('Loaded vocabulary from {}: {} words'.format(args.vocab, vocab_size))

    labels = {}
    if args.labels:
        with open(args.labels, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[parts[0]] = parts[1]

    preprocess(paths, word_to_index, vocab_size, args.output_dir, labels=labels, min_replies=args.min_replies)


if __name__ == '__main__':
    main()
