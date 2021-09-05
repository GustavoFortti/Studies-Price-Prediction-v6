import time
import argparse

from config.aplication import Config
from src.model import Model

def main(args):
    config = Config(currency=args.currency, question=int(args.question), _type=int(args.type_model)).config
    Model(config=config, mode=args.mode, index=int(args.index))
    
if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m" ,"--mode", help="tr - train data, te - test predict, pr - predict data, td - test data, gd - generate data", required=True)
    parser.add_argument("-t" ,"--type_model", help="1 = Classification, 2 = Regression", required=False)
    parser.add_argument("-i" ,"--index", required=False)
    parser.add_argument("-c" ,"--currency", required=False)
    parser.add_argument("-q" ,"--question", help="1 = Close, 2 = Open, 3 = High, 4 = Low, 5 = [High, Low]", required=False)
    args = parser.parse_args()
    if (not args.index): args.index = 0

    main(args)

    print("--- %s seconds main ---" % (time.time() - start_time))