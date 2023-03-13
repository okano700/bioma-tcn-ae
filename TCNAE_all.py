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

import gc


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

    begin = 0
    split_name = ds.name.split('_')
    if int(split_name[0]) < begin:
        continue
    print(f"{ds.name}", flush = True)
    period = get_period(np.array(ds.df['value'].loc[:ds.lenTrain]), 3)
    res = []
    for p in period:
        print(f"Running period{p}", flush = True)
        #loop for mult
        for m in [2,3,5]:
            for it in range(5):

                print(f"Running it {it}", flush = True)
                xTrain = ds.ts[:ds.lenTrain]
                xTest = ds.ts[ds.lenTrain:]
                scaler = MinMaxScaler()
                
                T = m*p
                if T%2 != 0:
                    T = T+1

                xTrain_scaled = scaler.fit_transform(xTrain.reshape(-1,1))

                xTrain_scaled = create_sequences(xTrain_scaled,T)
                tcn_ae = TCNAE(latent_sample_rate = 2)

                start_time = time.time()
                tcn_ae.fit(xTrain_scaled, xTrain_scaled, batch_size=32, epochs=5, verbose=1)
                print("> Time:", round(time.time() - start_time), "seconds.")

                test = scaler.transform(ds.df['value'].values.reshape(-1,1))[np.newaxis,:,:]
                anomaly_score = tcn_ae.predict(test)

                fpr, tpr, thresholds = roc_curve(ds.df['is_anomaly'].iloc[ds.lenTrain:len(anomaly_score)], anomaly_score[ds.lenTrain:len(anomaly_score)])
                res_auc = auc(fpr,tpr)
                res.append({'Dataset':ds.name, 'Period': p, 'Iteration': it, 'Multiplier': m, 'AUC': res_auc})
                pd.DataFrame(res).to_csv(f'res/TCNAE_AUC_UCR_{begin}.csv', index = False)
                np.savetxt(f'res/{ds.name}_{p}_{m}_{it}.npy')


                gc.collect()


