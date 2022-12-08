import numpy as np


class NeuralNetwork:
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    def __init__(self, layer_sizes, learning_rate):

        """
        this function initializes the neural netowrk and creates the random function (size based on parameters given)
        that is slowly refined by other functions.
        :param layer_sizes: these are the amount of nodes in each layer of the neural net
        :param learning_rate: this is an arbitrary number defined by the user to define how large of a change the
        neural net should make each process
        """

        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        # this creates the matrix that is the proper dimensions based on the layer sizes

        self.weights = [np.random.standard_normal(s) / s[1] ** .5 for s in weight_shapes]
        # fills neural net with random weights based on a standard distrubution

        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]  # generates biases for each weight
        self.learning_rate = learning_rate

    def neuron_output(self, training_images):

        """
        this function calculates the output of each neuron in sequence when its given data. using the output we can 
        start to train the neurons to output the correct answer
        :param training_images: 
        :return: void
        """

        tr_img = training_images
        self.output_list = []  # this initializes the array that stores the output neurons' weights and biases
        self.output_list.append(training_images)
        for i, j in zip(self.weights, self.biases):  # this loops through all the weights and biases in the net
            newWeights = np.matmul(i, tr_img)  # this generates the weight of one of the output neurons
            newBiases = np.add(newWeights, j)  # this does the same for biases
            activation = [self.activation(k) for k in newBiases]
            # the biases pass through the static activation function
            tr_img = activation
            self.output_list.append(activation)

    def o_error(self, training_labels):

        """
        this function calculates the output error of neurom output by comparing the nets answers with the datas given
        labels
        :param training_labels: the afformentioned data labels
        :return: void
        """

        self.output_error = np.subtract(training_labels, self.output_list[-1])

    def hidden_error(self):

        """
        this function calculates the error of the hidden layers of the net (those that are neither the input or output
        layers)by using whats called backpropagation starting from our output error and calculating the error of each
        preceding layer
        :return:
        """

        self.error = []
        self.error.append(self.output_error)
        out_error = self.output_error
        for i in self.weights[:0:-1]:
            weight_transpose = np.transpose(i)
            hidden_error = np.matmul(weight_transpose, out_error)
            out_error = hidden_error
            self.error.append(hidden_error)

    def d_weights(self):

        """
        this function calculates the change (delta) that should be applied to each node's weight as defined by their 
        error. this function uses gradient descent to calculate how much each weight should be changed
        :return: void
        """

        delta_weights = []
        Error = self.error
        Output = self.output_list
        for i, k, j, q in zip(Error[::-1], Output[1:], range(len(Output)), self.biases):
            sgmd_dvtv = [self.dactivation(a) for a in k]
            # finds the output for the derivative of a sigmoid fx w/ input as k
            error_sgmd_drvt = np.multiply(i, sgmd_dvtv)
            gradient = np.multiply(self.learning_rate, error_sgmd_drvt)
            gradient = np.add(gradient, q)
            output_transpose = np.transpose(Output[j])
            dlt_weit = np.matmul(gradient, output_transpose)
            delta_weights.append(dlt_weit)
        self.weights = np.add(self.weights, delta_weights)

    @staticmethod
    def activation(x):
        return (1 / (1 + np.exp(-x)))

    @staticmethod
    def dactivation(y):
        return (y * (1 - y))


# ----------------------------------------------------------------------------------------------------------------------#

def train(layer_sizes, learning_rate, training_length):
    with np.load('mnist.npz') as data:
        training_images = data['training_images']
        training_labels = data['training_labels']

        net = NeuralNetwork(layer_sizes, learning_rate)

    q = 0
    for i, j in zip(training_images, training_labels):
        net.neuron_output(i)
        if (np.argmax(net.output_list[-1])) == (np.argmax(j)):
            q = q + 1

    print('before training')
    print('{0}/50000 or {1}% correct'.format(q, (q / 50000) * 100))

    for i in range(training_length):
        for i, j in zip(training_images, training_labels):
            net.neuron_output(i)
            net.o_error(j)
            net.hidden_error()
            net.d_weights()

    k = 0
    for i, j in zip(training_images, training_labels):
        net.neuron_output(i)
        if (np.argmax(net.output_list[-1])) == (np.argmax(j)):
            k = k + 1

    var = []
    print('after training')
    print('{0}/50000 or {1}% correct'.format(k, (k / 50000) * 100))
    var.append((k / 50000) * 100)


# ----------------------------------------------------------------------------------------------------------------------#


def test(w):
    with np.load('mnist.npz') as data:
        import matplotlib.pyplot as plt
        test_images = data['test_images']
        test_labels = data['test_labels']
        print('this was the image the computer had to work with: ')
        plt.imshow(test_images[w].reshape(28, 28), cmap='gray')
        plt.show()
        print('it represents {}'.format(np.argmax(test_labels[w])))


# ----------------------------------------------------------------------------------------------------------------------#


def main():
    train((784, 5, 10), 0.1, 5)
    test(10)


# ----------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    main()
