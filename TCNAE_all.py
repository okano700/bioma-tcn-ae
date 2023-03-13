import pandas as pd
import numpy as np
from src.TSds import TSds
import tensorflow as tf
import sys
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import periodogram
from math import floor
import time
from src.tcnae import TCNAE
from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay, classification_report
import glob


def create_sequences(values, time_steps:int):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def get_period(data:np.array, n:int)-> list:
    f, px = periodogram(data, detrend='linear',nfft=int(len(data)*0.1) )
    p = []
    aux = 2
    for i in range(len(px)):
        #print(len(p))
        if len(p)>=n:
            break
        elif len(p) == 0:
            p.append(floor(1/f[np.argmax(px)] + 0.5))
        else:
            flag = False
            v = floor(1/f[px.argsort()[-aux]] + 0.5)
            for i in range(len(p)):
                
                if (p[i]%v != 0) and (v%p[i] != 0):
                    pass
                else:
                    flag = True
                    break
            if flag ==False:
                p.append(v)
            aux+=1
    return p


SEED = 42

tf.keras.utils.set_random_seed(SEED)

srcUCR = "/mnt/nfs/home/eyokano/datsets/UCR_Anomaly_FullData/"
UCR = [p for p in glob.glob(f"{srcUCR}/*.txt")]

print("Running UCR DS", flush = True)
res = []
res_f = []
UCR.sort()
b_s = 32
for path in UCR:
    ds = TSds.read_UCR(path)
    print(ds.name)
