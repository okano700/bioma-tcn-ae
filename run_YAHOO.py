import glob
import os

from utils.TSds import TSds
from utils.find_frequency import get_period
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str,required = True)
    parser.add_argument("--wl",type = int, required = False, default = 0)
    parser.add_argument("--i", type = int, required = False, default = 0)
    args = parser.parse_args()
    print(args)

    ds = glob.glob(args.path)
    ds.sort()
    for data in ds:
        ds = TSds.read_YAHOO(data)
        for period in get_period(ds.ts[:ds.train_split],3):
            for n in [2, 3, 5]:
                
                for i in range(5):
                    os.system(f"python3 ~/bioma-tcn-ae/run_TCNAE.py --path {data} --WL {period} --n {n} --i {i} --seed {i} >> log_TCNAE_YAHOO.log")

                    print(f"python3 ~/bioma-tcn-ae/run_TCNAE.py --path {data} --WL {period} --n {n} --i {i} --seed {i}")
