import os, sys, argparse
from pandas import DataFrame as df
import pandas as pd

print('aaa')
parser = argparse.ArgumentParser(description='Runing the experiment')
parser.add_argument('-b','--book_list', nargs='+', help='lists of query set from ...')
parser.add_argument('-f','--fragment_size', nargs='+', help='Number of fragments')
arg = parser.parse_args()

print(arg)
print(arg.book_list[1])


def this():
    a = 1
    b = 2
    c = 3
    return a, b, c


def df21212():
    f = [1, 2, 3]
    k = [0, -1, 99]
    tuples = []

    for ff in f:
        for kk in k:
            tuples += [(ff, kk)]

    index = pd.MultiIndex.from_tuples(tuples, names=['Fragments', 'KNearestNeighbor'])
    s = df(index=index, columns=("a","b","c"))
    s.loc[(1,-1), "a"] = 600
    print(s)


if __name__ == '__main__':
    (a, b, c) = this()
    print(a)
    print(b)
    print(c)
    df21212()



# load query documents: doc_id to query
    doc = []
    line = linecache.getlines("query_set_ids.csv")
    for i in range(len(line)):
        sliceline = line[i].replace("\n", "").replace("\r", "").split(",")
        doc.append(sliceline)
    querySet = []
    for i in range(len(doc)):
        for j in range(len(doc[i])):
            doc[i][j] = float(doc[i][j])
            querySet.append(doc[i][j])
    print querySet

    print "indexing"
    start = time.time()
    # extract labels
    doc_id = my_data[:, 0].astype(int)
    author_id = my_data[:, 2].astype(int)
    para_id = my_data[:, 3].astype(int)
    dataset = my_data[:, 4:].astype(float)
    # following line for old solution without set id's

    # for multilingua features
    # dataset = my_data[:,[8, 31, 32 ,33 ,34, 35, 36, 37, 39, 42, 43, 59]]

    # dataset = my_data[:,[7, 8, 9 , 10, 11, 12, 13, 14, 15, 30, 31 ,32 ,33, 34, 35, 36, 38, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]]

    # dataset = my_data[:,[3, 4, 5, 6, 7, 9, 12, 14, 15, 16, 17, 19, 20, 21, 24, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 38, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57]]
    print "dataset loaded properly"
    # dataset = my_data[:,[4, 5, 6, 7, 8, 10, 13, 15, 16, 17, 18, 20, 21, 22, 25, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 58]]

    # generate Indexes
    para_to_doc_dict = dict(zip(para_id, doc_id))
    doc_to_au_dict = dict(zip(doc_id, author_id))
    para_to_au_dict = dict(zip(para_id, author_id))
    paraIndex = dict()
    IndexPara = dict()
    for i in range(len(para_id)):
        paraIndex[para_id[i]] = i

    for i in range(len(para_id)):
        IndexPara[i] = para_id[i]

    doc_to_para_dict = defaultdict(list)
    for key, value in sorted(para_to_doc_dict.iteritems()):
        doc_to_para_dict[value].append(key)
    doc_to_para_dict = dict(doc_to_para_dict)

    au_to_para_dict = defaultdict(list)
    for key, value in sorted(para_to_au_dict.iteritems()):
        au_to_para_dict[value].append(key)
    au_to_para_dict = dict(au_to_para_dict)

    au_to_doc_dict = defaultdict(list)
    for key, value in sorted(doc_to_au_dict.iteritems()):
        au_to_doc_dict[value].append(key)
    au_to_doc_dict = dict(au_to_doc_dict)
    print time.time() - start, "seconds used to index"

    # normalize data
    start = time.time()
    # datasetm= NormalizeDataset(dataset)
    datasetm = multiprocessNorm(dataset)
    datasetQ = []
    datasetP = []
    datasetN = []
    print time.time() - start, "seconds used to normalize"
    for i in range(N):
        datasetN.append(datasetm[i])

    for i in range(NP):
        datasetP.append(datasetm[i])

    hitparaIndex = dict()
    hitIndexPara = dict()

    for doc in querySet:
        try:
            paraList = doc_to_para_dict[doc]
            for para in paraList:
                datasetQ.append(datasetm[paraIndex[para]])
            hitparaIndex[para] = (len(datasetQ) - 1)
            hitIndexPara[(len(datasetQ) - 1)] = para
        except KeyError:
            continue

    SplitN = SplitDataset(datasetN)
    SplitQ = SplitDataset(datasetQ)
    SplitP = SplitDataset(datasetP)



