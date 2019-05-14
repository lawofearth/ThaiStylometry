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
# import smjModule
import heapq
from scipy.spatial.distance import euclidean
import csv
from scipy import stats
import os, sys, argparse, shutil
from pandas import DataFrame as df
import pandas as pd
import extention, get_set_ids, first, results_compilation

parser = argparse.ArgumentParser(description='Runing the experiment')
parser.add_argument('-b','--book_list', nargs='+', help='lists of query set from ...')
parser.add_argument('-f','--fragment_size', nargs='+', help='Number of fragments')
parser.add_argument('-k','--topKNN', nargs='+', help='Number of K Nearest Neighbors')
parser.add_argument('-o','--output_name', nargs='+', help='Output file name')
arg = parser.parse_args()

print "=========Extention=========="
extention

print "=========Get_set_ids=========="
get_set_ids

print "=========First=========="
first

print "=========Result Compilation=========="
results_compilation