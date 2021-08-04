import time
import argparse

from src.model import Model

def main(args):
    Model(args.mode, int(args.index))

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m" ,"--mode", help="tr - train data | te - test predict | pr - predict data | td - test data | gd - generate data", required=True)
    parser.add_argument("-i" ,"--index", required=False)
    args = parser.parse_args()
    if (not args.index): args.index = 0

    main(args)

    print("--- %s seconds main ---" % (time.time() - start_time))