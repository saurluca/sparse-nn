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
    # Clip x to avoid overflow in exp()
    x = np.clip(x, -100, 100) # TODO this just temp fix, fix properly in network
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
        
        self.lr = 0.01 # learning rate
        # self.weight_decay = 0.99
        self.w_max = 2
        
        self.activation_fn = activation_fn
        self.final_activation_fn = sigmoid
        self.max_distance = n_neurons // 2
        print(f"max_distance {self.max_distance}")
        self.allow_self_recursion = False
        self.step_counter = 0

        self.prob_threshold = 0.2

        # activation of current time step
        self.activation = np.zeros(self.n_neurons)
        # previous activation
        self.prev_activation = np.zeros(self.n_neurons)

        assert input_dim + output_dim <= n_neurons

        # init neurons as 1D array of neurons, each having one array of weights with 0
        self.neurons = np.zeros((self.n_neurons, self.n_neurons))

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
        self.neuron_mask = np.array([[1 if w != 0 else 0 for w in weights] for weights in self.neurons])

        print("neuron mask")
        for mask in self.neuron_mask:
            print(mask)
        print()

    def print_network(self, simple=False):
        for idx, weights in enumerate(self.neurons):
            if simple:
                print([1 if w != 0 else 0 for w in weights])
            else:
                print(f"neuron {idx} weights: {weights}")

    def print_activation(self):
        print(f"Activation {self.activation}")

    def forward(self, x=None):
        next_activation = [0 for _ in range(self.n_neurons)]
        output = []

        # if there is an input
        if x is not None:
            # add input x to current actiavtion of input neurons
            for i in range(len(x)):
                self.activation[i] += x[i]

        # calculate step of for each neuron
        for i, neuron in enumerate(self.neurons):
            # sum up input of all incoming connections to neuron
            input_x = 0
            for w in range(self.n_neurons):
                if self.neuron_mask[i][w] == 1:
                    input_x += self.activation[w] * neuron[w]

            # append to output
            if i >= self.n_neurons - self.output_dim:
                output_x = self.final_activation_fn(input_x)
                output.append(output_x)
            else:
                output_x = self.activation_fn(input_x)
            next_activation[i] = output_x

        # read output
        self.step_counter += 1
        self.prev_activation = self.activation
        self.activation = next_activation
        self.output = output
        return output

    # def update(self):
    #     for neuron, activation in zip(self.neurons, self.activation):
    #         # calcualte hebbian learning update step
    #         update = self.lr * neuron * activation
    #         neuron += update
    #         # print(f"neuron: {neuron}, lr: {self.lr}, activation: {activation} update: {update}")
            
    #         # cap weight vector length
    #         neuron = neuron / np.maximum(1, np.linalg.norm(neuron) / self.w_max)
        
    def update(self):
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if self.neuron_mask[i][j] == 1:
                    # Proper Hebbian: weight change = lr * pre_synaptic * post_synaptic
                    self.neurons[i][j] += self.lr * self.activation[j] * self.activation[i]
            
            # Then normalize
            weight_norm = np.linalg.norm(self.neurons[i])
            if weight_norm > self.w_max:
                self.neurons[i] = self.neurons[i] * self.w_max / weight_norm

    # define forward as default method
    def __call__(self, x):
        return self.forward(x)


def gen_data(n=100):
    x = np.random.randint(2, size=(n, 2))
    labels = (x[:, 0] == x[:, 1]).astype(int)
    return zip(x, labels)


def train(data, model, single_sample=False):
    correct_samples = 0.0
    total_samples = 0
    for x, label in data:
        total_samples += 1
        # model.print_activation()
        y = model.forward(x)
        model.update()
        # model.print_activation()
        # for i in range(2):
        #     y = model.forward()
        # model.print_activation()

        output_class = np.round(y)
        if output_class == label:
            correct_samples += 1
        # print(f"raw: {np.round(y, 3)} output class: {output_class}, true class {label}")
        if single_sample:
            break

    if total_samples > 0:
        accuracy = correct_samples / total_samples
        print(f"final accuracy {accuracy}")
    else:
        print("No samples processed")


def main():
    data = gen_data(n=1000)

    model = SparseNN(input_dim=1, activation_fn=sigmoid, n_neurons=10)
    train(data, model, single_sample=False)


if __name__ == "__main__":
    main()
