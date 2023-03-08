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
    tf.config.list_physical_devices(,)

    gpus = tf.config.list_physical_devices(‘GPU’)

    tf.config.set_visible_devices(gpus[0], ‘GPU’)

    xTest = ds.ts[ds.lenTrain:]
    scaler = MinMaxScaler()

    xTrain_scaled = scaler.fit_transform(xTrain.reshape(-1,1))

    xTrain_scaled = create_sequences(xTrain_scaled,2 * get_period(xTrain, 1)[0])

    #print(xTrain_scaled.shape, len(xTrain_scaled))
    
    tcn_ae = TCNAE(latent_sample_rate = 2)
    start_time = time.time()
    tcn_ae.fit(xTrain_scaled, xTrain_scaled, batch_size=32, epochs=50, verbose=1)

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

