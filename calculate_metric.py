# Compute the distance between two stores based on the definition in the paper.

import pickle
import random
import numpy

f = open('feature_train_data.pickle', 'rb')
(X, y) = pickle.load(f)

dictlist = [{} for _ in range(1115)]
for feature, sale in zip(X, y):
    store = feature[1]
    dictlist[store][tuple(feature[2:7])] = sale

with open("embeddings.pickle", 'rb') as f:
    embeddings = pickle.load(f)
store_embeddings = embeddings[0]


def distance(store_pairs, dictlist):
    '''Distance as defined in the paper'''
    absdiffs = []
    a, b = store_pairs
    for key in dictlist[a]:
        if key in dictlist[b]:
            absdiffs.append(abs(dictlist[a][key] - dictlist[b][key]))
    return sum(absdiffs) / float(len(absdiffs))


def embed_distance(store_pairs, em):
    '''Distance in the embedding space'''
    a, b = store_pairs
    a_vec = em[a]
    b_vec = em[b]
    return(numpy.linalg.norm(a_vec - b_vec))

# Generate n random store pairs
n = 10000
pairs = set()
while len(pairs) < n:
    a, b = random.sample(range(1115), 2)
    if a < b:
        pairs.add((a, b))


# Calcuate distances
with open('distances.csv', 'w') as f:
    for pair in pairs:
        d = distance(pair, dictlist)
        d_em = embed_distance(pair, store_embeddings)
        print(d, d_em, file=f)
