import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model

from tensorflow.python.keras.layers import * #Masking, ConvLSTM2D, LSTM, Bidirectional, Dense, Dropout
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
 
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

        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(self.categorical, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'])

        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=8, shuffle=True, validation_data=(x_test, y_test), verbose=1)

    def regression(self, x_train, x_test, y_train, y_test) -> None:
        self.model = Sequential()

        self.model.add(Bidirectional(LSTM(32, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))))
        self.model.add(Dropout(0.5))

        self.model.add(Bidirectional(LSTM(32, return_sequences=False)))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(self.categorical, activation='softmax'))
        self.model.add(Dense(1))
        
        # RMSprop
        # adamax
        opt = tf.keras.optimizers.RMSprop(
            learning_rate=0.0001,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
        )

        self.model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=['cosine_similarity'])
        self.model.fit(x_train, y_train, epochs=5,
            # self.epochs, 
            batch_size=8, shuffle=True, validation_data=(x_test, y_test), verbose=1)

    def regression_1(self, x_train, x_test, y_train, y_test) -> None:
        self.model = Sequential()
        self.model.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = False, data_format = "channels_last", input_shape = (seq_len, img_height, img_width, 3)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(6, activation = "softmax"))
        opt = keras.optimizers.SGD(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    def regression_2(self, x_train, x_test, y_train, y_test) -> None:
        self.self.model = ConvLSTM2D(filters=16
                       , kernel_size=(3, 3)
                       , data_format='channels_first'
                       , recurrent_activation='hard_sigmoid'
                       , activation='tanh'
                       , padding='same'
                       , return_sequences=True
                       , input_shape=(x_train.shape[1], x_train.shape[2]))

        self.model = tf.keras.models.Sequential()

        self.model.add(Conv1D(filters=32, kernel_size=8, strides=1, activation="relu", padding="same",input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(MaxPooling1D(pool_size = 2))
        self.model.add(Conv1D(filters=16, kernel_size=8, strides=1, activation="relu", padding="same"))
        self.model.add(MaxPooling1D(pool_size = 2))
        
        
        self.model.add(Masking(mask_value=0.0))
        self.model.add(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
        self.model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
        self.model.add(Dense(30, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        
        self.model.add(Dense(num_classes, activation='softmax'))

    def regression_3(self, x_train, x_test, y_train, y_test) -> None:
        batch_size=68
        units=128
        learning_rate=0.005
        epochs=20
        dropout=0.2
        recurrent_dropout=0.2
        X_train = np.random.rand(700, 50,34)
        y_train = np.random.choice([0, 1], 700)
        X_test = np.random.rand(100, 50, 34)
        y_test = np.random.choice([0, 1], 100)
        loss = tf.losses.binary_crossentropy

        model = tf.keras.models.Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
        # uncomment the line beneath for convolution
        # model.add(Conv1D(filters=32, kernel_size=8, strides=1, activation="relu", padding="same"))
        model.add(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
        model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
        model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        adamopt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        model.compile(loss=loss,
                    optimizer=adamopt,
                    metrics=['accuracy'])

        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            verbose=1)

        score, acc = model.evaluate(X_test, y_test,
                                    batch_size=batch_size)

        yhat = model.predict(X_test)

    def save(self) -> None:
        self.model.save(self.path)
    
    def predict(self, x) -> np.array:
        model = tf.keras.models.load_model(self.path)
        return model.predict(x)
