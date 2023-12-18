from scipy.signal import periodogram
from math import floor
import numpy as np 
import argparse
from TSds import TSds


def get_period(data:np.array, n:int)-> list:
    f, px = periodogram(data, detrend='linear',nfft=int(len(data)*0.1) )
    p = []
    aux = 2
    for i in range(len(px)):
        #print(len(p))
        if len(p)>=n:
            break
        elif len(p) == 0:
            if f[np.argmax(px)] == 0:
                p.append(int(len(data)/10))
                return p
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type= str, required=True)
    parser.add_argument("--source", type =str, required=True)
    parser.add_argument("--n", type = int, default=1)

    args = parser.parse_args()

    if args.source == "UCR":
        data = TSds.read_UCR(args.path)
    elif args.source == "YAHOO":
        data = TSds.read_YAHOO(args.path)
    elif args.source == "NAB":
        data = TSds.read_NAB(args.path)


    print(get_period(data.ts, args.n))



