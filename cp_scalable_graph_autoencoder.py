import argparse
import time
import datetime
import os.path
import pickle
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
import networkx as nx
import dgl
import numpy as np
from tqdm import tqdm
from utils.data_utils import *
from utils.eval_utils import *
from utils.model_utils import *
from utils.models import *
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.sparse as sp
import re
from community.community_louvain import best_partition


MIDDLE = ""
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, help="Which dataset", default="100K")
parser.add_argument("-H", "--hidden", type=int, help="Hidden size", default=256)
parser.add_argument("-e", "--epoch", type=int, help="Max epochs", default=20)
parser.add_argument("-l", "--layers", type=int, help="N layers", default=1)
parser.add_argument("-he", "--heads", type=int, help="N heads", default=8)
parser.add_argument("-eo", "--eo_limit", type=float, help="Edge overlap limit", default=0.99)
# 连续多少个Epoch不上升则停止
parser.add_argument("-ep", "--ep_limit", type=int, help="Max epochs for eo ascending", default=30)
parser.add_argument("-s", "--seed", type=int, help="Random seed", default=50)
parser.add_argument("-re", "--remove", type=int, help="Remove self-loops", default=0)
parser.add_argument("-ud", "--undir", type=int, help="Undirected or not", default=0)
pargs = parser.parse_args()


def printPeakGPUMem():
    print('Peak GPU memory cached in training process: {}'.format(torch.cuda.max_memory_cached(device=args.device)))
    print('Peak GPU memory allocated in training process: {}'.format(torch.cuda.max_memory_allocated(device=args.device)))
    print('Peak GPU memory reserved in training process: {}'.format(torch.cuda.max_memory_reserved(device=args.device)))


def loadGraph(filename):
    arr = np.load(filename)
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    return sp.csr_matrix(arr)


def LoadStaticGraph(filename="/home/xuchenhao/datasets/CollegeMsg/CollegeMsg.txt",
                      save_path=None, remove_sl=False, undir=False):
    """
    读static graph文件名，输出static graph
    :param filename: 数据集文件名，默认为edgelist，header默认为None，每一行默认为(src, dst)
    :return: 保存了source node，target node， 时间戳的字典
    """
    static_dict = dict()
    static_dict["src"] = []
    static_dict["dst"] = []
    f = open(filename)
    tmp_data = f.readlines()
    f.close()
    lines = set()
    for line in tqdm(tmp_data):
        line = re.split('\s', line.strip())
        if ((not undir) or ((line[0], line[1]) not in lines and (line[1], line[0]) not in lines)):
            lines.add((line[0], line[1]))
            if ((not remove_sl) or (line[0] != line[1])):
                static_dict["src"].append(line[0])
                static_dict["dst"].append(line[1])
    node_set = set(static_dict["src"]+static_dict["dst"])
    node_name_to_idx = {node_name: i for i, node_name in enumerate(list(node_set))}
    src_names = pd.Series(static_dict.get("src"))
    dst_names = pd.Series(static_dict.get("dst"))
    src_names = src_names.apply(lambda x: node_name_to_idx[x])
    dst_names = dst_names.apply(lambda x: node_name_to_idx[x])
    static_dict["src"] = src_names.values
    static_dict["dst"] = dst_names.values
    if save_path is None:
        save_path = os.path.join("/home/xuchenhao/datasets/CollegeMsg/", "edgelist_date.csv")
    else:
        save_path = os.path.join(save_path, "edgelist.csv")
    pd.DataFrame(static_dict).to_csv(save_path, sep=" ", header=False, index=False)
    static_dict["node_to_idx"] = node_name_to_idx
    return static_dict


def FromStaticGraphToSparseAdj(dataset="CollegeMsg", save_path=None, remove_sl=False, undir=False):
    filename="/home/xuchenhao/datasets/" + MIDDLE + "{}/plain.txt".format(dataset, dataset)
    if save_path is None:
        save_path = "/home/xuchenhao/datasets/" + MIDDLE + "{}/".format(dataset)
    StaticGraph = LoadStaticGraph(filename=filename, save_path=save_path, remove_sl=remove_sl, undir=undir)
    node_num = len(StaticGraph["node_to_idx"])
    static_src = StaticGraph["src"]
    static_dst = StaticGraph["dst"]
    static_edges = (static_src, static_dst)
    dglg = dgl.graph(static_edges)
    save_path = os.path.join(save_path, "dgl_graph.bin")
    dgl.data.utils.save_graphs(save_path, [dglg])
    target_src = StaticGraph["src"]
    target_dst = StaticGraph["dst"]
    train_nids = np.unique(target_src)
    return sp.coo_matrix((np.ones(len(target_src)), (target_src, target_dst)),
                         shape=(node_num, node_num)), train_nids



DEVICE = "cuda:0"
EDGE_OVERLAP_LIMIT = {
    'CORA-ML' : 0.9,
}


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def param_compare(s1, s2, o):
    d1 = pd.read_csv(s1).drop(columns=["Unnamed: 0"])
    d2 = pd.read_csv(s2).drop(columns=["Unnamed: 0"])
    d3 = abs(d1 - d2) / d1
    s1 = d3.mean()
    s2 = d3.median()
    s1.name = 'mean'
    s2.name = 'median'
    d3 = d3.append(s1)
    d3 = d3.append(s2)
    d3.to_csv(o, index=False)


class ParseArguments(object):
    def __init__(self):
        self.data_name = pargs.dataset
        self.device = 'cuda:0'
        self.statistics_step = 10
        self.number_of_samples = 5
        self.n_layers = pargs.layers    # 1
        self.H = pargs.hidden           # 不同数据集不一样，DBLP 256、MSG 128、BTC-A 96、MATH 256，一般尽量取大（显存不够）
        self.n_heads = pargs.heads      # 8
        self.batch_size = 128
        self.g_type = 'temporal'                      
        self.lr = 0.0003
        self.weight_decay = 1e-6
        self.max_epochs = pargs.epoch   # 大多数数据集最后的结果取的800，少点比如500结果相差也不大，像UBUNTU这样特别慢的应该是跑了400
        self.graphic_mode = 'overlap'
        self.criterion = 'eo'
        self.eo_limit = pargs.eo_limit  # 0.99
        self.ep_limit = pargs.ep_limit  # 0.99
        self.seed = pargs.seed
        self.remove = True if pargs.remove else False
        self.undir = True if pargs.undir else False
        self.graph_path = './graphs/cp/{}_epo={}_lr={}_H={}_layers={}_heads={}_eo={}_s={}.npz'.format(self.data_name, self.max_epochs, self.lr, self.H, self.n_layers, self.n_heads, self.eo_limit, self.seed)  # path to save generated graphs
        self.table_path = './tables/cp/{}_epo={}_lr={}_H={}_layers={}_heads={}_eo={}_s={}.csv'.format(self.data_name, self.max_epochs, self.lr, self.H, self.n_layers, self.n_heads, self.eo_limit, self.seed)
        self.result_path = './results/cp/{}_epo={}_lr={}_H={}_layers={}_heads={}_eo={}_s={}.csv'.format(self.data_name, self.max_epochs, self.lr, self.H, self.n_layers, self.n_heads, self.eo_limit, self.seed)


args = ParseArguments()
if args.seed is not None:
    random_seed(args.seed)


if __name__ == '__main__':
    print('\n==================== Arguments ====================')
    print('Data Name: {}'.format(args.data_name))
    print('H: {}'.format(args.H))
    print('Max Epoches: {}'.format(args.max_epochs))
    print('N Layers: {}'.format(args.n_layers))
    print('N Heads: {}'.format(args.n_heads))
    label_adj, nids = FromStaticGraphToSparseAdj(args.data_name, remove_sl=args.remove, undir=args.undir)
    label_mat = label_adj.tocsr()[nids, :]
    num_nodes = label_adj.shape[1]
    t = label_adj.shape[0] // num_nodes
    feat = sp.diags(np.ones(num_nodes * t).astype(np.float32)).tocsr()
    adj = label_adj.tocsr()
    adj[adj > 1] = 1
    label_adj = adj.tocoo()
    g = nx.from_scipy_sparse_matrix(adj)
    tmp = best_partition(g)
    num_partitions = len(set(tmp.values()))
    comm_label = torch.LongTensor(list(tmp.values()))
    print('\n==================== Raw Graph Params ====================')
    print('Num Nodes: {}'.format(num_nodes))
    print('Num Edges: {}'.format(label_adj.nnz))
    print('Num Timstamps: {}'.format(t))
    sp.save_npz(open('./data/{}.npz'.format(pargs.dataset).format(), 'wb'), label_adj.tocsr())
    dgl_g = dgl.load_graphs(os.path.join("/home/xuchenhao/datasets/" + MIDDLE + "{}/".format(args.data_name), "dgl_graph.bin"))[0][0]
    dgl_g = dgl.add_self_loop(dgl_g)
    train_sampler = MultiLayerFullNeighborSampler(n_layers=args.n_layers)
    train_dataloader = NodeDataLoader(dgl_g.to(args.device),
                                      nids=torch.from_numpy(nids).long().to(args.device),
                                      block_sampler=train_sampler,
                                      device=args.device,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=0)
    encoder = GATLayer(in_dim=num_nodes,
                        hid_dim=int(args.H/args.n_heads),
                        n_heads=args.n_heads).to(args.device)
    comm_decoder = nn.Linear(args.n_heads * int(args.H/args.n_heads), num_partitions).to(args.device)
    decoder = nn.Linear(args.n_heads * int(args.H/args.n_heads), num_nodes).to(args.device)
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    comm_dec_opt = torch.optim.Adam(comm_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    max_eo = 0
    max_ep = 0
    start_time = time.time()
    print('\n==================== Train ====================')
    total_train_time = 0
    for epoch in range(args.max_epochs):
        num_edges_all = 0
        num_loss_all = 0
        encoder.train()
        comm_decoder.train()
        decoder.train()
        train_time = 0
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            batch_inputs, batch_labels, batch_comm_labels = coo_to_csp(feat[input_nodes.cpu(), :].tocoo()).to(args.device), \
                                                            coo_to_csp(adj[seeds.cpu(), :].tocoo()).to_dense().to(args.device), \
                                                            comm_label[seeds.cpu()].to(args.device)
            train_start_time = time.time()
            # train comm_decoder
            train_batch_clus_res = comm_decoder(encoder(blocks[0], batch_inputs))
            comm_loss = F.cross_entropy(train_batch_clus_res, batch_comm_labels)
            enc_opt.zero_grad()
            comm_dec_opt.zero_grad()
            comm_loss.backward()
            enc_opt.step()
            comm_dec_opt.step()
            # train decoder
            blocks = [block.to(args.device) for block in blocks]
            train_batch_logits = decoder(encoder(blocks[0], batch_inputs))
            num_edges = batch_labels.sum() / 2
            num_edges_all += num_edges
            loss = -0.5 * torch.sum(batch_labels * torch.log_softmax(train_batch_logits, dim=-1)) / num_edges
            num_loss_all += loss.cpu().data * num_edges
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            loss.backward()
            enc_opt.step()
            dec_opt.step()
            train_time += time.time() - train_start_time
            total_train_time += time.time() - train_start_time
            if (step+1) % 50 == 0:
                print("Epoch: {:03d}, Step: {:03d}, loss: {:.7f}".format(epoch+1, step+1, loss.cpu().data))
            else:
                sys.stdout.flush()
                sys.stdout.write("Epoch: {:03d}, Step: {:03d}, loss: {:.7f}\r".format(epoch+1, step+1, loss.cpu().data))
                sys.stdout.flush()
        print("Epoch: {:03d}, Overall Loss: {:.7f}, Time Consumption: {}s".format(epoch + 1, num_loss_all/num_edges_all, train_time))
        if (epoch+1) % 5 == 0:
            gen_mat = sp.csr_matrix(adj.shape)
            encoder.eval()
            comm_decoder.eval()
            decoder.eval()
            gen_time = 0
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                    gen_start_time = time.time()
                    test_inputs = coo_to_csp(feat[input_nodes.cpu(), :].tocoo()).to(args.device)
                    blocks = [block for block in blocks]
                    test_batch_logits = torch.softmax(decoder(encoder(blocks[0], test_inputs)), dim=-1)
                    num_edges = adj[seeds.cpu(), :].sum()
                    gen_mat[seeds.cpu(), :] = edge_from_scores(test_batch_logits.cpu().numpy(), num_edges)
                    gen_time += time.time() - gen_start_time
                    if (step+1) % 20 == 0:
                        print("Epoch: {:03d}, Generating Step: {:03d}".format(epoch+1, step+1))
                    else:
                        sys.stdout.flush()
                        sys.stdout.write("Epoch: {:03d}, Generating Step: {:03d}\r".format(epoch+1, step+1))
                        sys.stdout.flush()
            eo = adj.multiply(gen_mat).sum() / adj.sum()
            print("Epoch: {:03d}, Edge Overlap: {:07f}, Total Time: {}s, Generation Time: {}s, Total Time Consumption: {}s".format(epoch + 1, eo, int(time.time() - start_time), gen_time, total_train_time))
            if eo > max_eo:
                max_eo = eo
                max_ep = epoch + 1
                print("### New Best Edge Overlap: {:07f} ###".format(eo))
                with open(args.graph_path, 'wb') as f:
                    sp.save_npz(f, gen_mat)
                if eo > args.eo_limit:
                    break
            elif epoch + 1 >= max_ep + args.ep_limit:
                print("!!! Early Stopping after {} Epochs of EO Non-Ascending !!!".format(args.ep_limit))
                break
    printPeakGPUMem()
    # 释放显存
    torch.cuda.empty_cache()

    print('\n==================== Eval ====================')
    g = sp.csr_matrix((num_nodes, num_nodes))
    with open(args.graph_path, 'rb') as f:
        gen_mat = sp.load_npz(args.graph_path)
    stats = []
    for i in tqdm(range(t)):
        g += gen_mat[i * num_nodes: (i + 1) * num_nodes]
        g += g.T + sp.eye(g.shape[0])
        g[g > 1] = 1
        stats.append(compute_graph_statistics(g))
    stat_df = pd.DataFrame({k: [stats[s][k] for s in range(len(stats))] for k in stats[0].keys()})
    stat_df.to_csv(args.table_path)
    # raw_stats = pd.read_csv(args.raw_param_path).to_dict()
    # eps = np.finfo('float64').eps
    # comp_df = pd.DataFrame({k: [(abs(stats[s][k] - raw_stats[k][s]) / np.maximum(raw_stats[k][s], eps)) for s in range(len(stats))] for k in stats[0].keys()})
    # comp_df[comp_df > 1e10] = 0
    # s1 = comp_df.mean()
    # s2 = comp_df.median()
    # s1.name = 'mean'
    # s2.name = 'median'
    # comp_df = comp_df.append(s1)
    # comp_df = comp_df.append(s2)
    # comp_df.to_csv(args.result_path)