import time
import argparse

from src.model import Model
import pandas as pd

def main(args):
    mode = args.mode
    model = Model(args.mode)

    if (mode == 'tr'): model.train()
    if (mode == 'te'): model.test(args.index)
    if (mode == 'pr'): model.pred()

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m" ,"--mode", required=True)
    parser.add_argument("-i" ,"--index", required=False)
    args = parser.parse_args()
    main(args)
    print("--- %s seconds main ---" % (time.time() - start_time))