import gc
import warnings
from time import time
import os 

import networkx as nx
import pickle
from networkx.classes.function import subgraph
import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from networkx.algorithms.centrality import eigenvector_centrality

from layer import GCN
from utils import accuracy
from utils import macro_f1
from utils import CudaUse
from utils import EarlyStopping
from utils import LogResult
from utils import parameter_parser
from utils import preprocess_adj
from utils import print_graph_detail
from utils import read_file
from utils import return_seed

th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")


def get_train_test(target_fn):
    train_lst = list()
    test_lst = list()
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            if item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)

    return train_lst, test_lst


class PrepareData:
    def __init__(self, args):
        print("prepare data")
        self.graph_path = args.graph_path

        self.args = args

        # os.makedirs(args.data_path) 
        # os.makedirs(args.graph_path) 

        # graph
        graph = nx.read_weighted_edgelist(f"{self.graph_path}/{args.dataset}.txt"
                                          , nodetype=int)
        # print_graph_detail(graph)
        adj = nx.to_scipy_sparse_matrix(graph,
                                        nodelist=list(range(graph.number_of_nodes())),
                                        weight='weight',
                                        dtype=np.float)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = preprocess_adj(adj, is_sparse=True)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # features
        self.nfeat_dim = graph.number_of_nodes()
        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))

        # M. Amintoosi
        if self.args.use_gf == True:
            g_info = pickle.load(open(f"{self.graph_path}/{args.dataset}_g_info.pkl", 'rb'))
            #  زیرگراف حاصل از اسناد همبند نیست
            # subgraph = graph.subgraph(np.arange(g_info['num_docs']))

            # # زیر گراف حاصل از کلمات
            # subgraph = graph.subgraph(np.arange(g_info['num_docs'],self.nfeat_dim))
            # print('Is subgraph connected: ', nx.is_connected(subgraph))
            # # print_graph_detail(subgraph)
            ec_coef = 100
            # ec = eigenvector_centrality(subgraph)
            # dict1 = OrderedDict(sorted(ec.items()))
            # # value = list(dict1.values())

            bc = g_info["bc_subg"]
            dict1 = OrderedDict(sorted(bc.items()))

            # برای زیرگراف حاصل از اسناد
            # num_words = self.nfeat_dim-g_info['num_docs']
            # one_for_words = [1.] * num_words
            # value = [1+ec_coef*x for x in dict1.values()] + one_for_words

            # برای زیرگراف حاصل از کلمات
            num_docs = g_info['num_docs']
            one_for_docs = [1.] * num_docs
            value = one_for_docs + [1+ec_coef*x for x in dict1.values()]
            print("EC: ", value[-10:])
            # with open(f"{args.graph_path}/{args.dataset}_bc.pkl", 'wb') as outp:
            #     pickle.dump(bc, outp, pickle.HIGHEST_PROTOCOL)

        else:
            value = [1.] * self.nfeat_dim

        shape = (self.nfeat_dim, self.nfeat_dim)
        indices = th.from_numpy(
                np.vstack((row, col)).astype(np.int64))
        values = th.FloatTensor(value)
        shape = th.Size(shape)

        self.features = th.sparse.FloatTensor(indices, values, shape)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # target

        target_fn = f"{self.args.data_path}/{self.args.dataset}.txt"
        target = np.array(pd.read_csv(target_fn,
                                      sep="\t",
                                      header=None,
                                      engine='python')[2])
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [target2id[label] for label in target]
        self.nclass = len(target2id)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # train val test split

        self.train_lst, self.test_lst = get_train_test(target_fn)


class TextGCNTrainer:
    def __init__(self, args, model, pre_data):
        self.args = args
        self.model = model
        self.device = args.device

        self.max_epoch = self.args.max_epoch
        self.set_seed()

        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = EarlyStopping(args.early_stopping)

    def set_seed(self):
        th.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def fit(self):
        self.prepare_data()
        self.model = self.model(nfeat=self.nfeat_dim,
                                nhid=self.args.nhid,
                                nclass=self.nclass,
                                dropout=self.args.dropout)
        # print(self.model.parameters)
        self.model = self.model.to(self.device)

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = th.nn.CrossEntropyLoss()

        self.model_param = sum(param.numel() for param in self.model.parameters())
        # print('# model parameters:', self.model_param)
        self.convert_tensor()

        start = time()
        self.train()
        self.train_time = time() - start

    @classmethod
    def set_description(cls, desc):
        string = ""
        for key, value in desc.items():
            if isinstance(value, int):
                string += f"{key}:{value} "
            else:
                string += f"{key}:{value:.4f} "
        # print(string)

    def prepare_data(self):
        self.adj = self.predata.adj
        self.nfeat_dim = self.predata.nfeat_dim
        self.features = self.predata.features
        self.target = self.predata.target
        self.nclass = self.predata.nclass

        self.train_lst, self.val_lst = train_test_split(self.predata.train_lst,
                                                        test_size=self.args.val_ratio,
                                                        shuffle=True,
                                                        random_state=self.args.seed)
        self.test_lst = self.predata.test_lst

    def convert_tensor(self):
        self.model = self.model.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.target = th.tensor(self.target).long().to(self.device)
        self.train_lst = th.tensor(self.train_lst).long().to(self.device)
        self.val_lst = th.tensor(self.val_lst).long().to(self.device)

    def train(self):
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[self.train_lst],
                                  self.target[self.train_lst])

            loss.backward()
            self.optimizer.step()

            val_desc = self.val(self.val_lst)

            desc = dict(**{"epoch"     : epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)

            self.set_description(desc)

            if self.earlystopping(val_desc["val_loss"]):
                break

    @th.no_grad()
    def val(self, x, prefix="val"):
        self.model.eval()
        with th.no_grad():
            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[x],
                                  self.target[x])
            acc = accuracy(logits[x],
                           self.target[x])
            f1, precision, recall = macro_f1(logits[x],
                                             self.target[x],
                                             num_classes=self.nclass)

            desc = {
                f"{prefix}_loss": loss.item(),
                "acc"           : acc,
                "macro_f1"      : f1,
                "precision"     : precision,
                "recall"        : recall,
            }
        return desc

    @th.no_grad()
    def test(self):
        self.test_lst = th.tensor(self.test_lst).long().to(self.device)
        test_desc = self.val(self.test_lst, prefix="test")
        test_desc["train_time"] = self.train_time
        test_desc["model_param"] = self.model_param
        return test_desc


def main(dataset, times, use_gf=False):
    args = parameter_parser()
    args.dataset = dataset
    args.use_gf = use_gf

    args.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    args.nhid = 200
    args.max_epoch = 50
    args.dropout = 0.2
    args.val_ratio = 0.1
    args.early_stopping = 10
    args.lr = 0.02

    args.data_path = './data'
    args.tmp_path = '../tmp/TCGCN'
    args.graph_path = f"{args.tmp_path}/graph"

    model = GCN

    print(args)

    predata = PrepareData(args)
    cudause = CudaUse()

    record = LogResult()
    seed_lst = list()
    for ind, seed in enumerate(return_seed(times)):
        # print(f"\n\n==> {ind}, seed:{seed}")
        args.seed = seed
        seed_lst.append(seed)

        framework = TextGCNTrainer(model=model, args=args, pre_data=predata)
        framework.fit()

        if th.cuda.is_available():
            gpu_mem = cudause.gpu_mem_get(_id=0)
            record.log_single(key="gpu_mem", value=gpu_mem)

        record.log(framework.test())

        del framework
        gc.collect()

        if th.cuda.is_available():
            th.cuda.empty_cache()

    # print("==> seed set:")
    # print(seed_lst)
    record.show_str()


if __name__ == '__main__':
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    for d in ["mr", "ohsumed", "R52", "R8"]:#, "20ng"]:
        print("\n", d)
        main(d, 2)
        main(d, 2, use_gf=True)
    
    # main("mr", 1)
    # main("mr", 1, use_gf=True)

    # main("ohsumed",1)
    # main("R8", 5)
    # main("R8", 5, use_gf=True)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
