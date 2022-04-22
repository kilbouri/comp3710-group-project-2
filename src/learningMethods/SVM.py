from structures.SupportVectorMachine import SupportVectorMachineSK, SupportVectorMachineTF
from tensorflow import keras


class SVMLearnerSK():

    def __init__(self, attributes, className='classification') -> None:
        self.className = className
        self.attributes = attributes - {self.className}
        self.classifier = None

        # training config

        if len(self.attributes) == 0:
            raise ValueError(
                "Attribute list cannot be empty or contain only the classifier attribute!")

    def learn(self, data):
        # configure network construction process. The last addLayer call is the output layer.
        self.classifier = SupportVectorMachineSK(
            attributes=self.attributes,
            className=self.className
        )
        return self.classifier.learn(data)

    def evaluate(self, data):
        return self.classifier.predict(data)


class SVMLearnerKeras():

    def __init__(self, attributes, trainingEpoches=20, trainingBatch=128, validationSplit=0.8, className='classification') -> None:
        self.className = className
        self.attributes = attributes - {self.className}
        self.classifier = None

        # training config
        self.epochCount = trainingEpoches
        self.valSplit = validationSplit
        self.trainingBatch = trainingBatch

        if len(self.attributes) == 0:
            raise ValueError(
                "Attribute list cannot be empty or contain only the classifier attribute!")

    def learn(self, data):
        # configure network construction process. The last addLayer call is the output layer.
        self.classifier = SupportVectorMachineTF(
            attributes=self.attributes,
            trainingEpoches=self.epochCount,
            validationSplit=self.valSplit,
            trainingBatch=self.trainingBatch,
            className=self.className
        )\
            .addLayer(units=16, name='middle1', activation='relu')\
            .addLayer(units=4, name='middle2', activation='relu')\
            .addLayer(units=1, name='output', activation='sigmoid')
        self.classifier.create(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.hinge, metrics=[keras.metrics.binary_accuracy])
        return self.classifier.learn(data)

    def evaluate(self, data):
        return self.classifier.predict(data)
