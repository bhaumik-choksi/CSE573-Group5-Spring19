import sys
import os
import numpy as np
import pandas as pd
import torch
import aiohttp
import asyncio
import json
import requests
from utils import get_gpu_name, get_number_processors, get_gpu_memory, get_cuda_version
from parameters import *
from load_test import run_load_test

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)
print("Number of CPU processors: ", get_number_processors())
print("GPU: ", get_gpu_name())
print("GPU memory: ", get_gpu_memory())
print("CUDA: ", get_cuda_version())

nf_3m_valid = os.path.join(NF_DATA, 'N3M_VALID', 'n3m.valid.txt')
df = pd.read_csv(nf_3m_valid, names=['CustomerID','MovieID','Rating'], sep='\t')
print(df.shape)
print(df.head())

nf_3m_test = os.path.join(NF_DATA, 'N3M_TEST', 'n3m.test.txt')
df2 = pd.read_csv(nf_3m_test, names=['CustomerID','MovieID','Rating'], sep='\t')
print(df2.shape)
print(df2.head())

titles = pd.read_csv(MOVIE_TITLES, names=['MovieID'], encoding = "latin")
print(titles.head())

target = df2[df2['CustomerID'] == 0]
print(target)
df_customer = pd.merge(target, titles, on='MovieID', how='left', suffixes=('_',''))
print(df_customer)



df_query = df_customer.drop(['CustomerID','Title'], axis=1).set_index('MovieID')
dict_query = df_query.to_dict()['Rating']
print(dict_query)



