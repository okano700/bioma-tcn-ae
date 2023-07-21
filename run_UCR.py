import glob
import os

from numpy import require
from utils.TSds import TSds
from utils.find_frequency import get_period
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str,required = True)
    parser.add_argument("--n", type = int, required = False, default = 0)
    args = parser.parse_args()
    print(args)

    ds = glob.glob(args.path)
    ds.sort()

    for data in ds:
        spl = data.split('/')[-1]
        spl  = spl.split('_')[0]

        if int(spl) < args.n:
            continue

        ds = TSds.read_UCR(data)
        for period in get_period(ds.ts[:ds.train_split],3):
            for n in [2, 3, 5]:
                for i in range(5):
                    #os.system(f"python3 ~/bioma-tcn-ae/run_TCNAE.py --path {data} --WL {period} --n {n} --i {i} --seed {i}")

                    print(f"python3 ~/bioma-tcn-ae/run_TCNAE.py --path {data} --WL {period} --n {n} --i {i} --seed {i}")

