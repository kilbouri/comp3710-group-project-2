from structures.NeuralNetwork import NeuralNetwork
from itertools import product


class NNLearner():

    def __init__(self, attributes, trainingEpoches=5, trainingBatch=10, validationSplit=0.1, className='classification') -> None:
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
        )
        self.classifier.addLayer(32, name='Input', activation='relu', input_shape=[len(self.attributes)])
        self.classifier.addLayer(16, activation='relu')
        self.classifier.addLayer(4, activation='relu')
        self.classifier.addLayer(1, name='Output', activation='sigmoid')
        self.classifier.create(optimizer='adam', loss='binary_crossentropy')

        return self.classifier.learn(data)

    def evaluate(self, data):
        return self.classifier.predict(data)

    def tuneHyperparameters(self, data):
        batch_sizes = [5, 10, 20, 50, 75, 100]
        epochs = [5, 10, 20, 50, 75, 100]
        optimizers = ['RMSprop', 'adam', 'sgd']
        losses = ['MeanSquaredError', 'binary_crossentropy']
        kernel_initializers = [None, 'random_uniform']
        bias_initializers = [None, 'zeros']
        
        best = {"accuracy": 0, "loss": None,
            "optimizer": None, "batch_size": None, "epochs": None}
        with open("NNTrain.csv", "w") as f:
            f.write("Epochs,Batch size,Optimizer,Loss,Kernel,Bias,Accuracy\n")
        for optimizer, loss, kernel, bias, epoch, batch_size, in product(optimizers, losses, kernel_initializers, bias_initializers, epochs, batch_sizes):
            classifier = NeuralNetwork(
                attributes=self.attributes,
                trainingEpoches=epoch,
                validationSplit=self.valSplit,
                trainingBatch=batch_size,
                className=self.className
            )\
                .addLayer(32, name='Input', activation='relu', input_shape=[len(self.attributes)], kernel_init=kernel, bias_init=bias)\
                .addLayer(16, activation='relu', kernel_init=kernel, bias_init=bias)\
                .addLayer(4, activation='relu', kernel_init=kernel, bias_init=bias)\
                .addLayer(1, name='Output', activation='sigmoid', kernel_init=kernel, bias_init=bias)\
                .create(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            history = classifier.learn(data)
            avg_accuracy = history.history['accuracy'][-1]

            with open("NNTrain.csv", "a") as f:
                vals = [epoch, batch_size, optimizer, loss, kernel, bias, avg_accuracy*100]
                f.write(",".join(str(x) for x in vals) + "\n")
            print(f"Epochs: {epoch}, Batch size: {batch_size}, Optimizer: {optimizer}, Loss: {loss}, Kernel: {kernel}, Bias: {bias}, Accuracy: {avg_accuracy*100}")

            if(avg_accuracy > best["accuracy"]):
                best["accuracy"] = avg_accuracy
                best["loss"] = loss
                best["optimizer"] = optimizer
                best["batch_size"] = batch_size
                best["epochs"] = epoch
        with open('best.csv', 'w') as f:
            f.write('epochs,batch_size,optimizer,loss,kernel,bias,accuracy\n')
            f.write(f"{best['epochs']},{best['batch_size']},{best['optimizer']},{best['loss']},{best['accuracy']}")
        return best