import numpy as np

"""
TODO:
- make input neurons optionaly not have any other inputs/ only outputs/ vice versa output neurons
- vectorise
- def seperate pobabilty function for connections
- weight decay 
- hebbbian learning
- actual problem lol
"""


def identity(x):
    return x


def relu(x):
    return max(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SparseNN:
    """
    A Neural network in a 1 dimensional space. All neurons in one array.
    Neurons are sparsly connected. The closer 2 neurosn are in 1D space the higher the likelihood of being connected.

    Optionally: Neurons are not connected wiht themselves
    Potentially: lower weight for furhter neurons / weight decay
    """

    def __init__(self, input_dim, n_neurons, activation_fn, output_dim=1):
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.max_distance = n_neurons // 2
        print(f"max_distance {self.max_distance}")
        self.allow_self_recursion = False

        self.prob_threshold = 0.2

        # activation of current time step
        self.activation = [0 for _ in range(self.n_neurons)]

        assert input_dim + output_dim <= n_neurons

        # init neurons as 1D array of neurons, each having one array of weights with 0
        self.neurons = [
            [0 for _ in range(self.n_neurons)] for _ in range(self.n_neurons)
        ]

        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                # calculate distance between 2 neruons
                distance = np.abs(i - k)
                # print(f"i {i}, k {k}, distance {distance}")

                # check if a neurons is allowed to be connected with itself
                if not self.allow_self_recursion and distance == 0:
                    continue

                # calculate probability of neuron i having a conneciton to neuron k
                prob = np.random.random() / distance
                # if neurons are connected and in allowed max_distance
                if prob >= self.prob_threshold and distance <= self.max_distance:
                    self.neurons[i][k] = np.random.normal()

        # create 0/1 bool mask if a neuron connection has a weight or not
        self.neuron_mask = [
            [1 if w != 0 else 0 for w in weights] for weights in self.neurons
        ]

    def print_network(self, simple=False):
        for idx, weights in enumerate(self.neurons):
            if simple:
                print(self.neuron_mask)
            else:
                print(f"neuron {idx} weights: {weights}")

    def forward(self, x=None):
        next_activation = [0 for _ in range(self.n_neurons)]

        # if there is an input
        if x is not None:
            # add input x to current actiavtion of input neurons
            for i in range(len(x)):
                self.activation[i] += x[i]

        # calculate step of all neurons
        for i in range(self.n_neurons):
            # sum up input of all incoming connections to neuron
            input_x = 0
            for k in range(self.n_neurons):
                if self.neuron_mask[i][k] == 1:
                    input_x += self.activation[i][k] * self.neurons[i][k]

            output_x = self.activation_fn(input_x)

            next_activation[i][k] = output_x

        # read output

    # define forward as default method
    def __call__(self, x):
        return self.forward(x)


def main():
    model = SparseNN(input_dim=1, activation_fn=relu, n_neurons=5)
    model.print_network(simple=True)


if __name__ == "__main__":
    main()
