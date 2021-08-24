import numpy as np
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential

class LTSM_model():
    def __init__(self, config: dict) -> None:
        self.epochs = config.model['LSTM']['epochs']
        self.path = config.path + config.name + '/models/epochs_' + str(self.epochs) +  '_lstm_model.h5'
        self.categorical = config.data['target']['categorical']

    def classification(self, x_train, x_test, y_train, y_test) -> None:
        self.model = Sequential()

        self.model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.2))
        
        self.model.add(LSTM(32, return_sequences=False))
        self.model.add(Dropout(0.2))

        # fourth layer and output
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(self.categorical, activation='softmax'))

        # compile layers
        self.model.compile(loss='categorical_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'])

        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=8, shuffle=True, validation_data=(x_test, y_test), verbose=1)

    def regression(self, x_train, x_test, y_train, y_test) -> None:
        # Building the model 
        self.model = Sequential()
        # Adding a Bidirectional LSTM layer
        self.model.add(Bidirectional(LSTM(64,return_sequences=True, dropout=0.5, input_shape=(x_train.shape[1], x_train.shape[2]))))
        self.model.add(Bidirectional(LSTM(20, dropout=0.5)))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['mean_squared_error'])
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=8, shuffle=True, validation_data=(x_test, y_test), verbose=1)

    def save(self) -> None:
        self.model.save(self.path)
    
    def predict(self, x) -> np.array:
        model = tf.keras.models.load_model(self.path)
        return model.predict(x)
