import time
import argparse

from config.aplication import Config
from src.model import Model

from config.conf.Q1 import CONF as Q1 # close - loss: 0.5979 - accuracy: 0.6698 - val_loss: 0.6365 - val_accuracy: 0.6360
from config.conf.Q2 import CONF as Q2 # high low - loss: 0.9153 - accuracy: 0.5889 - val_loss: 0.9202 - val_accuracy: 0.5894
from config.conf.Q3 import CONF as Q3 # open - loss: 0.3844 - accuracy: 0.8276 - val_loss: 0.4483 - val_accuracy: 0.8020
from config.conf.Q4 import CONF as Q4 # high - loss: 0.5530 - accuracy: 0.7248 - val_loss: 0.5764 - val_accuracy: 0.6991
from config.conf.Q5 import CONF as Q5 # low - loss: 0.5291 - accuracy: 0.7335 - val_loss: 0.5788 - val_accuracy: 0.6953

def main(args):
    question = { 1: [Q1, "Q1"], 2: [Q2, "Q2"], 3: [Q3, "Q3"], 4: [Q4, "Q4"], 5: [Q5, "Q5"] }

    currency = args.currency
    name = currency + '_' + question[int(args.question)][1] + ("C" if (int(args.type_model) == 1) else "R")
    print(name)
    
    config = Config(currency=currency, name=name, config=question[int(args.question)][0], _type=int(args.type_model))
    Model(config=config, mode=args.mode, index=int(args.index))

    
if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m" ,"--mode", help="tr - train data | te - test predict | pr - predict data | td - test data | gd - generate data", required=True)
    parser.add_argument("-t" ,"--type_model", help="1 = Classification, 2 = Regression", required=False)
    parser.add_argument("-i" ,"--index", required=False)
    parser.add_argument("-c" ,"--currency", required=False)
    parser.add_argument("-q" ,"--question", required=False)
    args = parser.parse_args()
    if (not args.index): args.index = 0

    main(args)

    print("--- %s seconds main ---" % (time.time() - start_time))