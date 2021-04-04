import dgl
import ogb
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

device_id=0  # GPU 的使用 id
n_layers=3  # 输入层 + 隐藏层 + 输出层的数量
n_hiddens=256  # 隐藏层节点的数量
dropout=0.5
lr=0.01
epochs=300
runs=10  # 跑 10 次，取平均
log_steps=50

def train(model, g,feats,y_true,train_idx,optimizer):
    """
    训练函数
    :param model: 模型
    :param g: 图
    :param feats: 特征
    :param y_true: 标签
    :param train_idx: 下标
    :param optimizer: 优化器
    :return: 损失
    """
    model.train()
    optimizer.zero_grad()
    out = model(g, feats)[train_idx]
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, g, feats, y_true, split_idx, evaluator):
    """
    测试函数
    :param model: 模型
    :param g: 图
    :param feats: 特征
    :param y_true: 标签
    :param split_idx: 下标
    :param evaluator: 评价器
    :return: 结果
    """
    model.eval()

    out = model(g, feats)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


class Logger(object):
    """ 用于日志记录
    """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = DglNodePropPredDataset(name='ogbn-arxiv')
split_idx = dataset.get_idx_split()
g,labels = dataset[0]

feats = g.ndata['feat']
g = dgl.to_bidirected(g)
feats, labels = feats.to(device), labels.to(device)
train_idx = split_idx['train'].to(device)

from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self,in_feats,n_hiddens,n_classes,n_layers,dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(GraphConv(in_feats,n_hiddens,'both'))
        self.bns.append(nn.BatchNorm1d(n_hiddens))
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(n_hiddens, n_hiddens, 'both'))
            self.bns.append(nn.BatchNorm1d(n_hiddens))
        self.layers.append(GraphConv(n_hiddens, n_classes, 'both'))
        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, g, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(g,x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout,training=self.training)
        x = self.layers[-1](g,x)
        return x.log_softmax(dim=-1)

model = GCN(in_feats=feats.size(-1),n_hiddens=n_hiddens,n_classes=dataset.num_classes,
            n_layers=n_layers,dropout=dropout).to(device)

evaluator = Evaluator(name='ogbn-arxiv')
logger = Logger(runs)

for run in range(runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in range(1,1 + epochs):
        loss = train(model,g,feats,labels,train_idx,optimizer)
        result = test(model, g, feats, labels, split_idx, evaluator)
        logger.add_result(run,result)

        if epoch % log_steps == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
    logger.print_statistics(run)
logger.print_statistics()
