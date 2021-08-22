import time
import argparse

from config.aplication import Config
from src.model import Model

from config.conf.Q1 import CONF as Q1 # close - loss: 0.5979 - accuracy: 0.6698 - val_loss: 0.6365 - val_accuracy: 0.6360
from config.conf.Q2 import CONF as Q2 # high low - loss: 0.9153 - accuracy: 0.5889 - val_loss: 0.9202 - val_accuracy: 0.5894
from config.conf.Q3 import CONF as Q3 # open - loss: 0.3844 - accuracy: 0.8276 - val_loss: 0.4483 - val_accuracy: 0.8020
from config.conf.Q4 import CONF as Q4 # high - loss: 0.5530 - accuracy: 0.7248 - val_loss: 0.5764 - val_accuracy: 0.6991
from config.conf.Q5 import CONF as Q5 # low - loss: 0.5291 - accuracy: 0.7335 - val_loss: 0.5788 - val_accuracy: 0.6953
from config.conf.Q6 import CONF as Q6 # test

def main(args):
    currency = "XOM"
    name = currency + '_Q1'
    config = Config(currency, name, Q1)
    Model(config, args.mode, int(args.index))

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m" ,"--mode", help="tr - train data | te - test predict | pr - predict data | td - test data | gd - generate data", required=True)
    parser.add_argument("-i" ,"--index", required=False)
    parser.add_argument("-c" ,"--currency", required=False)
    args = parser.parse_args()
    if (not args.index): args.index = 0

    main(args)

    print("--- %s seconds main ---" % (time.time() - start_time))