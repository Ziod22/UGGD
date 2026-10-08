"""Create a BiGCN model, load trained weights, load a propagation-tree graph, and
produce a prediction.

Reference: TDrumorGCN/BUrumorGCN/Net below reproduce the model architecture from the
original BiGCN implementation -- https://github.com/TianBian95/BiGCN (Bian et al.,
"Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks",
AAAI 2020)

Usage:
    python load_bigcn.py --checkpoint checkpoints/BiGCN_data_iter0_fold4.pt \
        --input output/<id>.npz --dataset Twitter15
"""
import argparse
import copy

import numpy as np
import torch as th
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

LABELS = {
    'Twitter15': ['non-rumor', 'false', 'true', 'unverified'],
    'Twitter16': ['non-rumor', 'false', 'true', 'unverified'],
    'Weibo': ['false', 'true'],
}


# 1. Model

class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data, device):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data, device):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_classes):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 2, num_classes)

    def forward(self, data, device):
        TD_x = self.TDrumorGCN(data, device)
        BU_x = self.BUrumorGCN(data, device)
        x = th.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# 2. Load trained weights

def load_model(checkpoint_path, num_classes=4, in_feats=5000, hid_feats=64, out_feats=64, device=None):
    """Instantiate a BiGCN Net and load trained weights from checkpoint_path."""
    if device is None:
        device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    model = Net(in_feats, hid_feats, out_feats, num_classes).to(device)
    state_dict = th.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


# 3. Load data

def load_graph_from_npz(path):
    npz = np.load(path, allow_pickle=True)
    x = th.tensor(npz['x'], dtype=th.float32)
    edgeindex = npz['edgeindex']
    row, col = list(edgeindex[0]), list(edgeindex[1])
    edge_index = th.LongTensor([row, col])
    bu_edge_index = th.LongTensor([col, row])
    rootindex = th.LongTensor([int(npz['rootindex'])])
    batch = th.zeros(x.size(0), dtype=th.long)  # single graph -> all nodes in graph 0
    graph = Data(x=x, edge_index=edge_index, BU_edge_index=bu_edge_index, rootindex=rootindex)
    graph.batch = batch
    y = None
    if 'y' in npz.files:
        y = int(npz['y'])
    return graph, y


# 4. Produce a prediction

def predict(model, device, graph):
    graph = graph.to(device)
    with th.no_grad():
        log_probs = model(graph, device)
        probs = th.exp(log_probs).squeeze(0)
        pred_class = int(probs.argmax().item())
    return pred_class, probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='Path to a .pt/.m checkpoint saved by training')
    parser.add_argument('--input', required=True, help='.npz file for a single propagation-tree graph')
    parser.add_argument('--dataset', choices=list(LABELS.keys()), default='Twitter15',
                         help='Determines number of output classes and label names')
    args = parser.parse_args()

    num_classes = len(LABELS[args.dataset])
    model, device = load_model(args.checkpoint, num_classes=num_classes)
    graph, true_label = load_graph_from_npz(args.input)
    pred_class, probs = predict(model, device, graph)

    labels = LABELS[args.dataset]
    print("Predicted class: {} ({})".format(pred_class, labels[pred_class]))
    print("Class probabilities:")
    for label, p in zip(labels, probs):
        print("  {:>12s}: {:.4f}".format(label, p))
    if true_label is not None:
        print("Ground-truth label (from input file): {} ({})".format(true_label, labels[true_label]))


if __name__ == '__main__':
    main()
