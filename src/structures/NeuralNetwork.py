import os
#from attr import attr;

if os.getlogin() == 'Mat':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2" 

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from util import data_str_to_int

class NeuralNetwork:

    def __init__(self, attributes, validationSplit, trainingEpoches, trainingBatch, className='classification') -> None:
        """
        Creates a new neural network learning agent. Only the input layer is
        specified. You must use addLayer to add an appropriate output layer.
        """

        self.className = className
        self.attributes = attributes - {self.className}
        self.classifier = keras.Sequential()

        # training config
        self.epochCount = trainingEpoches
        self.valSplit = validationSplit
        self.trainingBatch = trainingBatch

        # builder safety: don't allow any changes after create() is called
        self.built = False

    def addLayer(self, units, activation=tf.nn.relu, name=None, kernel_init=None, bias_init=None, input_shape=None):
        if self.built:
            raise RuntimeError("Cannot add layers after building the network!")
        #hidden layer
        if input_shape is None:
            self.classifier.add(tf.keras.layers.Dense(
                name=name,
                units=units, 
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                activation=activation
            ))
        #input layer
        else:
            self.classifier.add(tf.keras.layers.Dense(
                name=name,
                units=units, 
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                activation=activation,
                input_shape=input_shape
            ))

        return self

    #def create(self, optimizer='sgd', loss='mse'):
    def create(self, optimizer='adam', loss='binary_crossentropy', metrics=None):
        if self.built:
            raise RuntimeError("Cannot build a network multiple times!")

        self.built = True
        self.classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return self

    def learn(self, data):
        if not self.built:
            raise RuntimeError("A neural network cannot learn before being built!")

        #convert into dataframe
        data_df = pd.DataFrame(data)
        data_df = data_str_to_int(data_df)

        # input training data, output training data (p = 0, e = 1)
        train_inputs, train_outputs = data_df.drop(columns=[self.className]), data_df[self.className]

        return self.classifier.fit(
            x=train_inputs,
            y=train_outputs,
            epochs=self.epochCount,
            batch_size=self.trainingBatch,
            validation_split=self.valSplit,
            verbose=0
        )

    def predict(self, data):
        if isinstance(data, dict):
            data_df = pd.DataFrame(data, index=[0])
            data_df = data_str_to_int(data_df)
            prediction = self.classifier.predict(data_df, batch_size=1, verbose=0)
            return 'e' if prediction[0][0] > 0.5 else 'p'
        else:
            data_df = pd.DataFrame(data)
            data_df = data_str_to_int(data_df)
            data_df.drop(columns=['classification'], inplace=True)
            prediction = self.classifier.predict(data_df)
            return list(map(lambda x: 'e' if x > 0.5 else 'p', prediction))