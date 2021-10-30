from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from math import floor
from sklearn.metrics import make_scorer, accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from keras.layers import LeakyReLU

score_acc = make_scorer(accuracy_score)
def nn_cl_bo2(neurons, activation, optimizer, learning_rate, batch_size, epochs, layers1, layers2, normalization, dropout, dropout_rate):

    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','SGD']
    optimizerD= {'Adam': Adam(learning_rate=learning_rate), 'SGD': SGD(learning_rate=learning_rate), 'RMSprop': RMSprop(learning_rate=learning_rate), 'Adadelta': Adadelta(learning_rate=learning_rate), 'Adagrad': Adagrad(learning_rate=learning_rate), 'Adamax': Adamax(learning_rate=learning_rate), 'Nadam': Nadam(learning_rate=learning_rate), 'Ftrl': Ftrl(learning_rate=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential', LeakyReLU, 'relu']

    neurons = round(neurons)
    activation = activationL[round(activation)]
    optimizer = optimizerD[optimizerL[round(optimizer)]]
    batch_size = round(batch_size)
    epochs = round(epochs)
    layers1 = round(layers1)
    layers2 = round(layers2)

    def nn_cl_fun():
        nn = Sequential()
        
        nn.add(LSTM(32, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]), activation=activation))
        nn.add(Dropout(0.20))
        nn.add(LSTM(32, return_sequences=False))
        nn.add(Dropout(0.20))
        if normalization > 0.5:
            nn.add(BatchNormalization())
        for i in range(layers1):
            nn.add(Dense(neurons, activation=activation))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate, seed=123))
        for i in range(layers2):
            nn.add(Dense(neurons, activation=activation))
        nn.add(Dense(1, activation='sigmoid'))
        nn.compile(loss='mse', optimizer=optimizer, metrics=['mean_absolute_percentage_error'])
        return nn

    y_arr = [i[0] for i in y_train]
    es = EarlyStopping(monitor='mean_absolute_percentage_error', mode='max', verbose=0, patience=20)
    nn = KerasRegressor(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    score = cross_val_score(nn, x_train, y_arr, scoring=score_acc, cv=kfold, fit_params={'callbacks':[es]}).mean()

    return score

params_nn2 = {
    'neurons': (10, 100),
    'activation':(0, 9),
    'optimizer':(0,7),
    'learning_rate':(0.01, 1),
    'batch_size':(200, 1000),
    'epochs':(20, 100),
    'layers1':(1,3),
    'layers2':(1,3),
    'normalization':(0,1),
    'dropout':(0,1),
    'dropout_rate':(0,0.3)
}

# Run Bayesian Optimization
nn_bo = BayesianOptimization(nn_cl_bo2, params_nn2, random_state=111)
nn_bo.maximize(init_points=25, n_iter=4)

# params_nn_ = nn_bo.max['params']
# learning_rate = params_nn_['learning_rate']
# activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
#             'elu', 'exponential', LeakyReLU,'relu']
# params_nn_['activation'] = activationL[round(params_nn_['activation'])]
# params_nn_['batch_size'] = round(params_nn_['batch_size'])
# params_nn_['epochs'] = round(params_nn_['epochs'])
# params_nn_['layers1'] = round(params_nn_['layers1'])
# params_nn_['layers2'] = round(params_nn_['layers2'])
# params_nn_['neurons'] = round(params_nn_['neurons'])
# optimizerL = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','Adam']
# optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
#             'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
#             'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
#             'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}
# params_nn_['optimizer'] = optimizerD[optimizerL[round(params_nn_['optimizer'])]]
# params_nn_
