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


def get_res(loss, df, m_v):

    # Generate AUC
    fpr, tpr, thresholds = roc_curve(df['is_anomaly'].iloc[m_v:len(loss)], loss.loc[m_v:len(loss)])
    res_auc = auc(fpr,tpr)
    res.append({'Dataset':ds_name, 'Period': p, 'Iteration': it, 'Multiplier': m, 'AUC': res_auc})
    pd.DataFrame(res).to_csv(f'res/TCNAE_AUC_UCR_{begin}.csv', index = False)

    # Generate f-score
    for tr in [0.01, 0.02, 0.03,0.05, 0.07,0.1,0.2,0.3,0.5]:
        resf = classification_report(df['anomaly'].iloc[m_v:len(loss)],[1 if it> tr else 0 for it in loss.loc[m_v:len(loss)] ],output_dict= True, target_names = ['normal','anomaly'], zero_division = 0)
        res_f.append({'Dataset':ds_name,
                            'Window_length':w_l,
                            'Multiplier': m,
                            'threshold': tr,
                            'Period': p,
                            'precision':resf['anomaly']['precision'],
                            'recall':resf['anomaly']['recall'],
                            'f1-score':resf['anomaly']['precision'],
                            'acc':resf['accuracy']})
        pd.DataFrame(res_f).to_csv(f'TCNAE_fscore_{begin}.csv', index = False)



def runModel(path:str, source:str, verbose:int):
    print(path,source, verbose)
    if source == 'UCR':
        ds = TSds.read_UCR(path)
    elif source == 'YAHOO':
        ds = TSds.read_YAHOO(path)
    elif source == 'NAB':
        ds = TSds.read_NAB(path)
    else:
        print('Invalid Source')
        return 0


    xTrain = ds.ts[:ds.lenTrain]
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    device_name = tf.test.gpu_device_name()
    print(device_name)
    tf.config.list_physical_devices()

    gpus = tf.config.list_physical_devices(‘GPU’)

    tf.config.set_visible_devices(gpus[0], ‘GPU’)

    xTest = ds.ts[ds.lenTrain:]
    scaler = MinMaxScaler()

    xTrain_scaled = scaler.fit_transform(xTrain.reshape(-1,1))

    xTrain_scaled = create_sequences(xTrain_scaled,2 * get_period(xTrain, 1)[0])

    #print(xTrain_scaled.shape, len(xTrain_scaled))
    
    tcn_ae = TCNAE(latent_sample_rate = 2)
    start_time = time.time()
    tcn_ae.fit(xTrain_scaled, xTrain_scaled, batch_size=32, epochs=2, verbose=1)

    print("> Training Time:", round(time.time() - start_time), "seconds.")
    start_time = time.time()
    test = scaler.trasform(ds.ts.reshape(-1,1))
    anomaly_score = tcn_ae.predict(test[numpy.newaxis,:,:])
    print("> Time:", round(time.time() - start_time), "seconds.")



if __name__ =='__main__':
    args = {}
    for index, arg in enumerate(sys.argv):
        if arg == '--path':
            args['path'] = sys.argv[index + 1]
        elif arg =='--source':
            args['source'] = sys.argv[index+1]
        elif arg =='--verbose':
            args['verbose'] = int(sys.argv[index + 1])
    runModel(**args)

