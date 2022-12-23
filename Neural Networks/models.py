from keras.layers import *
from keras.models import Model
import tensorflow as tf
import numpy as np 
from keras import Sequential
from keras.optimizers import adam_v2

class NeuralModels:
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.vocab_size = vocab_size
        self.embedding_matrix = emb_mat
        self.model = Sequential()

class CBOW(NeuralModels):
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        super().__init__(emb_mat, vocab_size, loss, epochs, optimizer, metrics)
    
    def fit(self,xtrain, xval, ytrain, yval):
        self.model.add(Dense(300, input_shape = (900,), activation = "sigmoid"))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(200, activation = "sigmoid"))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(100, activation = "sigmoid"))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(2, activation = "softmax"))
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)

    def get_model_summary(self):
        self.model.summary()
    
    def predict(self, xtest):
        return self.model.predict(xtest)

class LsTM(NeuralModels):
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        super().__init__(emb_mat, vocab_size, loss, epochs, optimizer, metrics)
    
    def train_model(self):
        inp1 = Input(shape = (128,))
        inp2 = Input(shape = (128,))
        emb1 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=128)(inp1)
        emb2 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=128)(inp2)
        concat = Concatenate(axis = -1)([emb1 + emb2, emb1 - emb2, emb1 * emb2])
        lstm = LSTM(150, return_sequences=False, dropout=0.1, return_state=True)(concat)
        out = Dense(2, activation = "softmax")(lstm[2])
        self.model = Model(inputs = [inp1, inp2], outputs = out)
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)

    def get_model_summary(self):
        self.model.summary()
    
    def predict(self, xtest):
        return self.model.predict(xtest)

class BiLSTM(NeuralModels):
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        super().__init__(emb_mat, vocab_size, loss, epochs, optimizer, metrics)
    
    def train_model(self):
        inp1 = Input(shape = (128,))
        inp2 = Input(shape = (128,))
        emb1 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=128)(inp1)
        emb2 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=128)(inp2)
        concat = Concatenate(axis = -1)([emb1 + emb2, emb1 - emb2, emb1 * emb2])
        out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, kernel_regularizer='l2', dropout=0.1, return_sequences=True))(concat)
        out = tf.keras.backend.mean(out, axis=1, keepdims=False)
        output = tf.keras.layers.Dense(2, kernel_regularizer='l2', activation='softmax')(out)
        self.model = Model(inputs = [inp1, inp2], outputs = output)
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)

    def get_model_summary(self):
        self.model.summary()
    
    def predict(self, xtest):
        return self.model.predict(xtest)

class LsTM_Attention(NeuralModels):
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        super().__init__(emb_mat, vocab_size, loss, epochs, optimizer, metrics)
    
    def train_model(self):
        inp1 = Input(shape = (128,))
        inp2 = Input(shape = (128,))
        emb1 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=128)(inp1)
        emb2 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=128)(inp2)
        
        lstm1 = LSTM(150, return_sequences=True, dropout=0.1, return_state=True)(emb1)
        lstm2 = LSTM(150, return_sequences=True, dropout=0.1, return_state=True)(emb2)

        attention = dot([lstm1[0], lstm2[0]], axes=[2, 2])
        u_norm = Softmax(axis=-1)(attention)
        v_norm = Softmax(axis=1)(attention)


        u = dot([u_norm, lstm1[0]], axes=[1, 1])
        v = dot([v_norm, lstm2[0]], axes=[1, 1])

        WU_bar = Dense(150)(u[:, -1, :])
        WV_bar = Dense(150)(v[:, -1, :])
        VU = Dense(150)(lstm1[0][:, -1, :])
        VV = Dense(150)(lstm2[0][:, -1, :])

        ufinal = Add()([WU_bar, VU])
        vfinal = Add()([WV_bar, VV])

        ufinal = Activation('tanh')(ufinal)
        vfinal = Activation('tanh')(vfinal)
        
        concat = Concatenate(axis = -1)([ufinal, vfinal])
        out = Dense(2, activation = "softmax")(concat)
        self.model = Model(inputs = [inp1, inp2], outputs = out)
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)


    def get_model_summary(self):
        self.model.summary()
    
    def predict(self, xtest):
        return self.model.predict(xtest)