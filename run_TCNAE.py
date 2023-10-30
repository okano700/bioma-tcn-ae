from src.tcnae import TCNAE
from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay, classification_report
from utils.TSds import TSds
from utils.find_frequency import get_period
from utils.scorer import scorer
import tensorflow as tf
import argparse
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def create_sequences(values, time_steps:int):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

if __name__ =="__main__":
    
    
    print(tf.config.list_physical_devices("GPU"))
    print(tf.test.is_built_with_cuda())
    #read the parser


    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type = str, required=True)
    parser.add_argument('--WL',type=int, required=True)
    parser.add_argument('--n',type=int, required=True)
    parser.add_argument('--i', type=int, required = True)
    parser.add_argument('--seed', type=int, required=True)


    args = parser.parse_args()

    print(args)

    SEQ_LEN = args.WL * args.n #if (args.WL * args.n)%2 == 0 else (args.WL * args.n) + 1

    tf.random.set_seed(args.seed)
    
    #Leitura dos dados
    ds = TSds.read_UCR(args.path)

    xTrain = ds.ts[:ds.train_split]
    xTest = ds.ts[ds.train_split:]
    scaler = MinMaxScaler()

    scaler.fit(xTrain.reshape(-1,1))

    xTrain_scaled = scaler.transform(xTrain.reshape(-1,1))
    xTest_scaled = scaler.transform(xTest.reshape(-1,1))

    xTrain_scaled = create_sequences(xTrain_scaled,SEQ_LEN)
    
    #Treino TCNAE
    tcn_ae = TCNAE(latent_sample_rate = args.WL, use_early_stopping=True)
    tcn_ae.fit(xTrain_scaled, xTrain_scaled, batch_size=32, epochs=50, verbose=1)

    test = scaler.transform(ds.df['value'].values.reshape(-1,1))[np.newaxis,:,:]

    anomaly_score = tcn_ae.predict(test)

    #Tratamento anomaly_score
    anomaly_score = np.append(anomaly_score,[0 for i in range(len(anomaly_score),len(ds.ts))])

    



    _, _, res = scorer(ds.df['is_anomaly'].loc[ds.train_split:],anomaly_score[ds.train_split:])
    
    res['dataset'] = ds.name
    res['WL'] = args.WL 
    res['n'] = args.n 
    res['id'] = args.i 

    path_to_res = "res_TCNAE_UCR.csv"

    if os.path.exists(path_to_res):
        print('existe')
        df = pd.read_csv(path_to_res)
        pd.concat([df, pd.DataFrame([res])]).to_csv(path_to_res, index = False)
    
    else: 
        pd.DataFrame([res]).to_csv(path_to_res, index = False)


    print(res)

