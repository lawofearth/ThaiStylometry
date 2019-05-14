from __future__ import division
import csv
import os

print(__doc__)
import time
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
import os
from scipy import stats
from scipy import spatial 
import operator 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets
from sklearn.metrics import metrics,accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
#import entropy as ent

#path = '/home/grads/raheem/Journal/icdm_shortDataset_Baseline/expResult10K_feature'
replacements = {'[':'', '(':'', ']':'',')':''}


shd_partial=[]
mhd_partial=[]
phd_partial=[]

shd_partial_k5=[]
mhd_partial_k5=[]
phd_partial_k5=[]

shd_partial_k7=[]
mhd_partial_k7=[]
phd_partial_k7=[]

shd_partial_k3=[]
mhd_partial_k3=[]
phd_partial_k3=[]

shd_partial_k1=[]
mhd_partial_k1=[]
phd_partial_k1=[]

shd_complete=[]
mhd_complete=[]
phd_complete=[]



pr_shd=[]
pr_mhd=[]
pr_phd=[]

exec_time_LSH_SHD=[]
exec_time_LSH_MHD=[]
exec_time_LSH_PHD=[]

average=[]


doc_ids_list=[]
author_ids_list=[]
phd_authors_list=[]

training_data=[]
# For execution time and pruning ratio use the other excel files
#following two lines will be used to calculate entropy based accuracy
#my_data =   loadtxt('C:/Users/rsarwar2/Desktop/multilingual_corpus/query_doc_ids_short.csv',delimiter=',')
#my_data=ent.compute_entropy(my_data)
# the following lines will be used if we want to calculate the overall accuracy of all subsets    
# my_data =   loadtxt('C:/Users/rsarwar2/Desktop/english corpus short/l.csv')
# my_data=ent.compute_entropy(my_data)
#my_data=my_data.tolist()
#my_data=[int(i) for i in my_data]
#my_data=[str(i) for i in my_data]
# set_ids= my_data[:,0]
# print (my_data[0])
# file_list=[]
# for filename in os.listdir('C:/Users/rsarwar2/Desktop/new set formulation/expResult10K_feature'):
    # file_list.append(file_name)
my_data=[]

for filename in os.listdir('./result'):    
    my_data.append(filename)
for filename in my_data:
    #print(float(filename))
    #print(float(filename) in my_data)
    #if (True==float(filename) in my_data):
    #    print("Type Error")
    with open("./result/"+filename) as infile, open("./expResult10KK/"+filename+filename, 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, target)
            outfile.write(line)

            

    results = []
    with open("./expResult10KK/"+filename+filename) as inputfile:
        for row in csv.reader(inputfile):
            results.append(row)       

# Partial Accuracy           
# ////////////////////////////////////////////////////////////////            
    for i in range(0,len(results[16])):
        if(results[16][i].strip()==results[15][0].strip()):
            shd_partial.append(1)
			
    for i in range(0,len(results[17])):
        if(results[17][i].strip()==results[15][0].strip()):
            mhd_partial.append(1)
			
    for i in range(0,len(results[18])):
        if(results[18][i].strip()==results[15][0].strip()):
            phd_partial.append(1)
            
            
# /////////////////////////////////////////////////////////			
# Complete Accuracy
# /////////////////////////////////////////////////////////			
    if(results[16][0].strip()==results[15][0].strip()):
        shd_complete.append(1)
			
    
    if(results[17][0].strip()==results[15][0].strip()):
        mhd_complete.append(1)
			
    
    if(results[18][0].strip()==results[15][0].strip()):
        phd_complete.append(1)		
     
    pr_shd.append((float(len(results[9]))/float(results[8][0].strip())*100))
    pr_mhd.append((float(results[10][0].strip())/float(results[8][0].strip())*100))
    pr_phd.append((float(results[11][0].strip())/float(results[8][0].strip())*100))   
# ////////////////////////////////////////////////////////////////////////////////////////			
        
# Execution Time calculations



#    exec_time_LSH_SHD.append(float(results[1][0].strip())+float(results[3][0].strip()))
#    exec_time_LSH_MHD.append(float(results[4][0].strip())+float(results[5][0].strip()))
#    exec_time_LSH_PHD.append(float(results[6][0].strip())+float(results[7][0].strip()))
    


# Get the number of candidate authors


    # doc_id = results[0][0].strip()
    # doc_ids_list.append(results[0][0].strip())
    # author_id= results[15][0].strip()
    # author_ids_list.append(author_id)
    # phd_authors =[]
    # i=0
    # for j in range(0,len(results[18])):
        # if(i>=len(results[18])):
            # break
        # phd_authors.append(results[18][i].strip())
        # i=i+2

    # phd_authors=set(phd_authors)
 
    # phd_authors=list(phd_authors)
    # phd_authors = [float(i) for i in phd_authors]
    

    
   # tt.main(doc_id,author_id,phd_authors) 



# doc_ids_list=[]
# author_ids_list=[]
# phd_authors_list=[]


    doc_id = results[0][0].strip()
    doc_ids_list.append(doc_id)
    author_id= results[15][0].strip()
    author_ids_list.append(author_id)
    phd_authors =[]
    i=0
    for j in range(0,len(results[18])):
        if(i>=len(results[18])):
            break
        phd_authors.append(results[18][i].strip())
        i=i+2
    phd_authors_list.append(phd_authors)
   
#    print (doc_id, author_id,phd_authors)

rows=zip(doc_ids_list,author_ids_list,phd_authors_list)
 
phd_authors = [float(i) for i in phd_authors]
# with open("large_candidate_authors.csv", "w") as f:
    # writer = csv.writer(f)#,dialect='excel',delimiter=' ')
    # for row in rows:
        # writer.writerow(row)
        #bla =[doc_id,author_id,phd_authors]
        #writer.writerow(bla)





#rint("If K is  10:")
#rint("Weak Accuracy SHD:", len(shd_partial)/len(my_data)*100)
#rint("Weak Accuracy MHD:", len(mhd_partial)/len(my_data)*100)
#rint("Weak Accuracy PHD:",len(phd_partial)/len(my_data)*100)
print("")
print("")
print("")
print("")
print("")

print("Strong Accuracy SHD:", len(shd_complete)/len(shd_partial)*100)
print("Strong Accuracy MHD:", len(mhd_complete)/len(mhd_partial)*100)
print("Strong Accuracy PHD", len(phd_complete)/len(phd_partial)*100)
print("")
print("")
print("")
print("")
#rint("Pruning Ratio SHD", sum(pr_shd)/len(my_data))
#rint("Pruning Ratio MHD", sum(pr_mhd)/len(my_data))
#rint("Pruning Ratio PHD", sum(pr_phd)/len(my_data))

print("")
print("")
print("")
print("")
#rint("Execution Time:")
#rint("Execution Time LSH-SHD", sum(exec_time_LSH_SHD)/len(my_data))
#rint("Execution Time LSH-MHD", sum(exec_time_LSH_MHD)/len(my_data))
#rint("Execution Time LSH-PHD", sum(exec_time_LSH_PHD)/len(my_data))

#rint("Average number of candidate authors:", sum(average)/float(len(average)))



