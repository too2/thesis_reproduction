from pyrwr.rwr import RWR
import pandas as pd
import numpy as np

class NRP():
    def __init__(self):
        self.seed = 10
        self.c = 0.15
        self.epsilon = 1e-9
        self.max_iters = 10

    def rwr_topk(self, seed, c, epsilon, max_iters, input_graph, graph_type='undirected'):
        '''
        computing a single source RWR score vector w.r.t. a given query (seed) node
        '''
        self.seed = seed
        self.c = c
        self.epsilon = epsilon
        self.max_iters = max_iters
        rwr = RWR()
        rwr.read_graph(input_graph, graph_type)
        r = rwr.compute(self.seed, self.c, self.epsilon, self.max_iters)
        return r


data_home = 'D:/BaiduNetdiskWorkspace/论文复现存放数据位置/PageRank/tutorial-on-link-analysis-master/data/small'
nrp = NRP()

label_path = "{}/node_labels.tsv".format(data_home)
node_labels = pd.read_csv(label_path, sep="\t")
num_nodes = len(set(node_labels.iloc[:,0]))
input_graph = data_home+'/edges.tsv'
# get the top-k of each node
ans_nrp = {}
for seed in np.arange(0, num_nodes):
    r = nrp.rwr_topk(seed = seed, c = 0.15, epsilon = 1e-9, max_iters = 10, input_graph=input_graph, graph_type='undirected')
    ans_nrp[seed] = r

ans = {}
def get_topk(k, ans_nrp):
    for i in ans_nrp.items():
        an = pd.DataFrame(i[1])
        an = an.sort_values(by = [0], ascending=False)
        for ind in np.arange(0, k+1):
            if ind == 0:
                ans[i[0]] = []
            if i[0] == an[an[0] == an.iloc[ind][0]].index.tolist()[0]:
                continue
            ans[i[0]].append((an[an[0] == an.iloc[ind][0]].index.tolist()[0], an.iloc[ind][0]))
get_topk(5, ans_nrp)
print(ans)

# 松弛标记
Pr = np.zeros((num_nodes, 1))
labels = pd.read_csv(data_home+'/node_label_zidingyi_ceshi.csv')
labels

for i in labels['id']:
    if labels['gender'][i] == ' male':
        Pr[i] = 0
    elif labels['gender'][i] == ' female':
        Pr[i] = 1
    else:
        Pr[i] = 0.5

def is_convergence(a, b):
    '''
    Judge whether the result converges
    Args:
        a (numpy.ndarray)
        b (numpy.ndarray)
    Returns:
        0：not converged
        1: converged
    '''
    ans = sum(abs(a-b))
    if ans < 0.01:
        return 1
    else:
        return 0

beita = 0.8
aerfa = 0.5

import copy
iter_times = 100
Pr1 = copy.deepcopy(Pr)
for t in np.arange(0, iter_times):
    for key_node, p in ans.items():
        Z = 0
        tmp = 0
        for j in p:
            tmp += Pr[j[0]] * j[1]
            Z += j[1]
        Pr1[key_node] = beita/Z*tmp + (1-beita)*Pr[key_node]
    print('###########第{}次迭代#############'.format(t+1))
    print(Pr1)
    if is_convergence(Pr, Pr1) == 1:
        print("#########已收敛，共迭代{}次########".format(t+1))
        break
    Pr, Pr1 = Pr1, copy.deepcopy(Pr)
    beita *= aerfa
print(Pr)
