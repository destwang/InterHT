# coding:utf-8

import sys
from ogb.linkproppred import LinkPropPredDataset
from collections import Counter
import pickle

dataset = LinkPropPredDataset('ogbl-wikikg2')

train = dataset.get_edge_split()['train']

heads = set(train['head'])
tail = set(train['tail'])
rels = set(train['relation'])

nodes = heads | tail
nentity = dataset.graph['num_nodes']

counter = Counter(train['head'])
counter.update(train['tail'])

rel_counter = Counter(train['relation'])

node_dict = {
    i: {
        'nbs': [],
        'rels': [],
        'rev_nbs': [],
        'rev_rels': []
    }
    for i in range(nentity)
}
for i in range(len(train['head'])):
    head_node = train['head'][i]
    tail_node = train['tail'][i]
    rel = train['relation'][i]
    node_dict[head_node]['nbs'].append(tail_node)
    node_dict[head_node]['rels'].append(rel)
    node_dict[tail_node]['rev_nbs'].append(head_node)
    node_dict[tail_node]['rev_rels'].append(rel)

for node in node_dict:
    nbs = node_dict[node]['nbs']
    rels = node_dict[node]['rels']
    if nbs:
        nbs_rels = list(zip(nbs, rels))
        nbs_rels = sorted(nbs_rels, key=lambda x: counter[x[0]], reverse=True)
        nbs, rels = list(zip(*nbs_rels))
        node_dict[node]['nbs'] = nbs
        node_dict[node]['rels'] = rels
        node_dict[node]['sort_rels'] = sorted(rels, key=lambda x: rel_counter[x])

    nbs = node_dict[node]['rev_nbs']
    rels = node_dict[node]['rev_rels']
    if nbs:
        nbs_rels = list(zip(nbs, rels))
        nbs_rels = sorted(nbs_rels, key=lambda x: counter[x[0]], reverse=True)
        nbs, rels = list(zip(*nbs_rels))
        node_dict[node]['rev_nbs'] = nbs
        node_dict[node]['rev_rels'] = rels
        node_dict[node]['sort_rev_rels'] = sorted(rels, key=lambda x: rel_counter[x])

node_set = set()
for node in node_dict:
    nbs = node_dict[node]['nbs'][:8]
    node_set |= set(nbs)
    nbs = node_dict[node]['rev_nbs'][:4]
    node_set |= set(nbs)
print(len(node_set))

node_list = list(node_set)
node_list = sorted(node_list)
node_mapping = {node: i for i, node in enumerate(node_list)}

for node in node_dict:
    nbs = node_dict[node]['nbs'][:8]
    node_dict[node]['nbs'] = [node_mapping[n] for n in nbs]
    nbs = node_dict[node]['rev_nbs'][:4]
    node_dict[node]['rev_nbs'] = [node_mapping[n] for n in nbs]

with open('data/node_digpiece_neighbors.pkl', 'wb') as f:
    pickle.dump(node_dict, f)
