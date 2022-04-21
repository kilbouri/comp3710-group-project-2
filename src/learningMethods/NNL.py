from structures.NeuralNetwork import NeuralNetwork


class NNLearner():

    def __init__(self, attributes, trainingEpoches=10, trainingBatch=10, validationSplit=0.1, className='classification') -> None:
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
        self.classifier = NeuralNetwork(
            attributes=self.attributes,
            trainingEpoches=self.epochCount,
            validationSplit=self.valSplit,
            trainingBatch=self.trainingBatch,
            className=self.className
        )\
            .addLayer(32, name='Input', activation='relu', input_shape=[len(self.attributes)])\
            .addLayer(16, activation='relu')\
            .addLayer(4, activation='relu')\
            .addLayer(1, name='Output', activation='sigmoid')\
            .create(optimizer='adam', loss='binary_crossentropy')

        return self.classifier.learn(data)

    def evaluate(self, data):
        return self.classifier.predict(data)

    def tuneHyperparameters(self, data):
        batch_sizes = [5, 10, 20, 50, 75, 100]
        epochs = [5, 10, 20, 50, 75, 100]
        optimizers = ['adam', 'sgd']
        losses = ['binary_crossentropy']

        best = {"accuracy": 0, "loss": None,
            "optimizer": None, "batch_size": None, "epochs": None}
        for batch_size in batch_sizes:
            for epoch in epochs:
                for optimizer in optimizers:
                    for loss in losses:
                        classifier = NeuralNetwork(
                            attributes=self.attributes,
                            trainingEpoches=epoch,
                            validationSplit=self.valSplit,
                            trainingBatch=batch_size,
                            className=self.className
                        )\
                            .addLayer(32, name='Input', activation='relu', input_shape=[len(self.attributes)])\
                            .addLayer(16, activation='relu')\
                            .addLayer(4, activation='relu')\
                            .addLayer(1, name='Output', activation='sigmoid')\
                            .create(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                        print("Batch size: " + str(batch_size) + " Epochs: " + str(epoch) + " Optimizer: " + str(optimizer) + " Loss: " + str(loss), end="")
                        history = classifier.learn(data)
                        avg_accuracy = history.history['accuracy'][-1]
                        print(f" = Accuracy: {100 * avg_accuracy:.2f}")
                        if(avg_accuracy > best["accuracy"]):
                            best["accuracy"] = avg_accuracy
                            best["loss"] = loss
                            best["optimizer"] = optimizer
                            best["batch_size"] = batch_size
                            best["epochs"] = epoch
        return best