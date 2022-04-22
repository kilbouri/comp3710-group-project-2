from sklearn import svm
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from util import issac_to_dfdict

class SupportVectorMachineSK:
    def __init__(self, attributes, className='classification'):
        self.className = className
        self.attributes = attributes - {self.className}

        self.classifier = svm.SVC(
            kernel='linear', decision_function_shape='ovo')

        # builder safety: don't allow any changes after create() is called
        self.built = False
        pass

    def learn(self, data):
        if self.built:
            raise RuntimeError("Cannot build a network multiple times!")
        self.built = True
        data_df = pd.DataFrame(issac_to_dfdict(data))
        data_df = data_df.apply(lambda s: s.astype('float32'))
        train_inputs, train_outputs = data_df.drop(columns=[self.className]), data_df[self.className]
        # input training data, output training data (p = 0, e = 1)

        return self.classifier.fit(train_inputs, train_outputs)

    def predict(self, data):
        data_df = pd.DataFrame(issac_to_dfdict(data))
        data_df = data_df.apply(lambda s: s.astype('float32'))
        test_inputs, test_outputs = data_df.drop(columns=[self.className]), data_df[self.className]
        prediction = self.classifier.predict(test_inputs)
        if isinstance(data, dict):
            return 'e' if prediction[0] == 1 else 'p'
        else:
            return list(map(lambda x: 'e' if x > 0.5 else 'p', prediction))


class SupportVectorMachineTF:
    def __init__(self, attributes, validationSplit, trainingEpoches, trainingBatch, className='classification'):
        self.className = className
        self.attributes = attributes - {self.className}

        # training config
        self.epochCount = trainingEpoches
        self.valSplit = validationSplit
        self.trainingBatch = trainingBatch

        self.classifier = keras.Sequential(
            [
                keras.Input(shape=(len(self.attributes),),
                            dtype='float32', name='input'),
                # RandomFourierFeatures(
                #     output_dim=4096, scale=10.0, kernel_initializer="gaussian", name='random_fourier'
                # ),
                #layers.Dense(units=16, name='middle1', activation='relu'),
                #layers.Dense(units=4, name='middle2', activation='relu'),
                #layers.Dense(units=1, name='output', activation='sigmoid'),
            ]
        )
        # builder safety: don't allow any changes after create() is called
        self.built = False

    def create(self, optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.hinge, metrics=[keras.metrics.binary_accuracy]):
        if self.built:
            raise RuntimeError("Cannot build a network multiple times!")
        self.built = True
        self.classifier.compile(
            optimizer=optimizer,
            loss=loss,
            # loss=keras.losses.binary_crossentropy,
            # metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
            metrics=metrics,
        )
        return self

    def addLayer(self, units, activation='relu', name=None, kernel_init='random_uniform'):
        if self.built:
            raise RuntimeError("Cannot add layers after building the network!")
        self.classifier.add(layers.Dense(
            name=name,
            units=units, 
            kernel_initializer=kernel_init,
            activation=activation
        ))
        return self

    def learn(self, data):
        if not self.built:
            self.create()
        # convert into dataframe and split
        data = pd.DataFrame(issac_to_dfdict(data))
        data = data.apply(lambda s: s.astype('float32'))
        train_inputs, train_outputs = data.drop(columns=[self.className]), data[self.className]

        # train the model
        # convert all values to float32 for vector multiplication

        # train the model
        self.classifier.fit(
            train_inputs,
            train_outputs,
            epochs=self.epochCount,
            batch_size=self.trainingBatch,
            validation_split=self.valSplit,
            verbose=0
        )
        return self

    def predict(self, data):
        data_df = pd.DataFrame(issac_to_dfdict(data))
        data_df = data_df.apply(lambda s: s.astype('float32'))
        if isinstance(data, dict):
            prediction = self.classifier.predict(data_df, batch_size=1)
            return 'e' if prediction[0][0] > 0.5 else 'p'
        else:
            data_df.drop(columns=['classification'], inplace=True)
            prediction = self.classifier.predict(data_df)
            return list(map(lambda x: 'e' if x > 0.5 else 'p', prediction))