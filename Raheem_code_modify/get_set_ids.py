from __future__ import division
print(__doc__)
import time
import itertools
import numpy as np
#from sklearn import preprocessing
#from sklearn.datasets.samples_generator import make_blobs
from collections import defaultdict
import math
import random
import multiprocessing
from numpy import genfromtxt
from numpy import loadtxt
import linecache
#import smjModule
import heapq
from scipy.spatial.distance import euclidean
import csv
from scipy import stats
#import os






my_data = loadtxt('./raheem.csv', delimiter=',')
my_data=np.asarray(my_data)

# ry documents: doc_id to query
doc = []
line = linecache.getlines("crosslingual_queryset.csv")
for i in range(len(line)):
    sliceline = line[i].replace("\n", "").replace("\r", "").split(",")
    doc.append(sliceline)
querySet = []
for i in range(len(doc)):
    for j in range(len(doc[i])):
        doc[i][j] = float(doc[i][j])
        querySet.append(int(doc[i][j]))
print querySet



q_doc_ids = querySet
my_data=my_data[np.logical_or.reduce([my_data[:,1] == x for x in q_doc_ids])]



labels=my_data[:,0:2]
labels=np.vstack({tuple(row) for row in labels})
# for x in labels:

    # print(x)
    
with open("query_set_ids.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(labels)


# with open("set_doc_ids.csv", "w") as text_file:
    # text_file.write("labels: %s\n" % labels)

#set_label=set(labels)
#set_label=list(set_label)
#print (set_label)
#with open('set_labels.csv', 'w') as fp:
#    a = csv.writer(fp, delimiter=',')
#    a.writerows(x)
    
