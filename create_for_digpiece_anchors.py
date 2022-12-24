from collections import defaultdict
from ogb.linkproppred import LinkPropPredDataset
import pickle
from tqdm import tqdm
from collections import Counter

neighbor_file = 'data/node_digpiece_anchors.pkl'

dataset = LinkPropPredDataset(name='ogbl-wikikg2')
num_nodes = dataset.graph['num_nodes']
num_rels = dataset.graph['edge_reltype'].max() + 1
split_dict = dataset.get_edge_split()
train_triples = split_dict['train']

nbors = defaultdict(set)
edges = defaultdict(set)
degrees = defaultdict(int)
for h, r, t in zip(train_triples['head'], train_triples['relation'],
                   train_triples['tail']):
    nbors[h].add(t)
    nbors[t].add(h)
    degrees[h] += 1
    degrees[t] += 1
    edges[(h, t)] = r

anchor_file = 'data/ogbl-wikikg2_20000_anchors_20000_paths_d0.4_b0.0_p0.4_r0.2_pykeen_50sp_bfs.pkl'
anchors, rels, vocab = pickle.load(open(anchor_file, "rb"))
sample_size = 20
ancs_set = set(anchors)
for e in tqdm(range(num_nodes)):
    anc_dists = list(zip(vocab[e]['ancs'], vocab[e]['dists']))
    anc_dists = sorted(anc_dists, key=lambda x: (x[1], degrees[x[0]]))

    subgraph_anchors = set()
    onehotanchors = ancs_set & nbors[e]
    non_anc_nbors = nbors[e] - onehotanchors
    counter = Counter()
    for node in non_anc_nbors:
        counter.update(nbors[node] & ancs_set)
    hop2_node_items = sorted(list(counter.items()),
                             key=lambda x: (-x[1], degrees[x[0]]))
    if hop2_node_items:
        hop2_nodes = list(zip(*hop2_node_items))[0]
    else:
        hop2_nodes = []
    hop1_nodes = sorted(list(onehotanchors), key=lambda x: degrees[x])
    ancs = hop1_nodes + list(hop2_nodes)

    other_hop2_nodes = set()
    for node in nbors[e]:
        other_hop2_nodes |= nbors[node] & ancs_set
    other_hop2_nodes = list(other_hop2_nodes - set(ancs))
    other_hop2_nodes = sorted(other_hop2_nodes, key=lambda x: degrees[x])
    ancs = ancs + other_hop2_nodes

    for node in ancs:
        subgraph_anchors.add(node)
        if len(subgraph_anchors) >= sample_size:
            break
    else:
        if len(vocab[e]
               ['dists']) > sample_size and vocab[e]['dists'][sample_size] > 2:
            hop1_set = nbors[e]
            hop2_set = set()
            for node in hop1_set:
                hop2_set |= nbors[node]
            hop2_set = hop2_set - hop1_set - ancs_set
            counter = Counter()
            for node in hop2_set:
                counter.update(nbors[node] & ancs_set)
            hop3_node_items = sorted(list(counter.items()),
                                     key=lambda x: (-x[1], degrees[x[0]]))
            if hop3_node_items:
                hop3_nodes = list(zip(*hop3_node_items))[0]
            else:
                hop3_nodes = []
            for node in hop3_nodes:
                subgraph_anchors.add(node)
                if len(subgraph_anchors) >= sample_size:
                    break

    if len(subgraph_anchors) < sample_size:
        for ee, d in anc_dists:
            subgraph_anchors.add(ee)
            if len(subgraph_anchors) >= sample_size:
                break
    vocab[e]['ancs'] = list(subgraph_anchors)

pickle.dump([anchors, rels, vocab], open(neighbor_file, "wb"))
