from numpy import loadtxt
import os
import numpy as np
import csv
import linecache


def get_set_ids(docu_namebu = '01.csv'):
    my_data = loadtxt('./raheem.csv', delimiter=',')
    my_data = np.asarray(my_data)

    # ry documents: doc_id to query
    doc = []
    line = linecache.getlines(docu_namebu)
    for i in range(len(line)):
        sliceline = line[i].replace("\n", "").replace("\r", "").split(",")
        doc.append(sliceline)
    querySet = []

    for i in range(len(doc)):
        for j in range(len(doc[i])):
            doc[i][j] = float(doc[i][j])
            querySet.append(int(doc[i][j]))

    q_doc_ids = querySet

    print q_doc_ids

    my_data = my_data[np.logical_or.reduce([my_data[:, 1] == x for x in q_doc_ids])]

    labels = my_data[:, 0:2]
    labels = np.vstack(sorted({tuple(row) for row in labels}))

    if 'query_set_idsw.csv' in os.listdir('./'):
        print('delete query set ids')

        os.remove('query_set_idsw.csv')

    print(os.listdir('./'))

    with open("query_set_idsw.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(labels)


get_set_ids('01.csv')