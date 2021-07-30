import time
import argparse

from src.model import Model
# from src.services.api import Api_trade

def main(args):
    mode = args.mode
    model = Model(mode)

    if (mode == 'td'): model.data()
    # if (mode == 'gd'): model.data(gen_data=True)
    # if (mode == 'tr'): model.train()

    # trade = Api_trade()
    # if ('--te'): 
    #     df_trade = model.test(init=0, end=10)
    #     trade.test(df_trade)
    # if ('--pr'): 
    #     df_trade = model.pred()
    #     trade.play(df_trade)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m" ,"--mode", required=True)
    args = parser.parse_args()
    main(args)
    print("--- %s seconds main ---" % (time.time() - start_time))