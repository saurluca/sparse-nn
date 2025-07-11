import numpy as np

"""
TODO:
- make input neurons optionaly not have any other inputs/ only outputs/ vice versa output neurons
- vectorise
- def seperate pobabilty function for connections
- weight decay 
- hebbbian learning
- actual problem lol

Later:
- in 2D create columsn with inter and intra connections
"""


def identity(x):
    return x


def relu(x):
    return max(0, x)


def sigmoid(x):
    # Clip x to avoid overflow in exp()
    x = np.clip(x, -100, 100)  # TODO this just temp fix, fix properly in network
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


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

        self.lr = 0.1  # learning rate
        self.weight_decay = 0.9
        self.w_max = 10

        self.activation_fn = activation_fn
        self.final_activation_fn = sigmoid
        self.max_distance = n_neurons // 2
        # self.max_distance = 10
        print(f"max_distance {self.max_distance}")
        self.allow_self_recursion = False
        self.step_counter = 0

        self.prob_threshold = 0.0

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
                    self.neurons[i][k] = np.random.normal(0, 0.5)

        # create 0/1 bool mask if a neuron connection has a weight or not
        self.neuron_mask = np.array(
            [[1 if w != 0 else 0 for w in weights] for weights in self.neurons]
        )

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
        
    def print_weight_ratio(self):
        # prints ratio of positive weights to negative weights
        positive_weights = 0
        negative_weights = 0
        for neuron in self.neurons:
            for weight in neuron:
                if weight > 0:
                    positive_weights += 1
                elif weight < 0:
                    negative_weights += 1
        print(f"positive weights: {positive_weights}, negative weights: {negative_weights}")
        if negative_weights > 0:
            print(f"ratio: {positive_weights / negative_weights:.2f}")
        else:
            print("no negative weights")
        print(f"total weights: {positive_weights + negative_weights}, positive weights: {positive_weights}, negative weights: {negative_weights}")

    def print_average_weight(self):
        # prints out hte average weight of all weights
        average_weight = np.mean(self.neurons)
        print(f"average weight: {average_weight}")
        
    def forward(self, x=None):
        # Reset activations instead of accumulating
        self.prev_activation = self.activation.copy()
        self.activation = np.zeros(self.n_neurons)
        next_activation = [0 for _ in range(self.n_neurons)]
        output = []

        # if there is an input
        if x is not None:
            # add input x to current actiavtion of input neurons
            # TODO changeable betwene setting and adding activation
            for i in range(len(x)):
                self.activation[i] = x[i]  # Set instead of add

        # calculate step of for each neuron
        for i, neuron in enumerate(self.neurons):
            # sum up input of all incoming connections to neuron
            input_x = 0
            for w in range(self.n_neurons):
                if self.neuron_mask[i][w] == 1:
                    input_x += self.prev_activation[w] * neuron[w]

            # append to output
            if i >= self.n_neurons - self.output_dim:
                output_x = self.final_activation_fn(input_x)
                # print(f"output_x {output_x} input_x {input_x}")
                output.append(output_x)
            else:
                output_x = self.activation_fn(input_x)
            next_activation[i] = output_x

        # read output
        self.step_counter += 1
        self.activation = next_activation
        self.output = output
        return output

    def update(self):
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if self.neuron_mask[i][j] == 1:
                    # Oja's rule: weight change = lr * post_synaptic * (pre_synaptic - post_synaptic * weight)
                    self.neurons[i][j] += (
                        self.lr * self.activation[i] * (self.activation[j] - self.activation[i] * self.neurons[i][j])
                    )
                    # Apply weight decay
                    self.neurons[i][j] *= self.weight_decay            

    # define forward as default method
    def __call__(self, x):
        return self.forward(x)


def gen_data(n=100):
    x = np.zeros((n, 2))
    labels = np.zeros(n)
    
    # Generate first column of random floats
    x[:, 0] = np.random.random(n)
    
    # For 50% of cases, make second column same as first
    same_indices = np.random.choice(n, size=int(0.5*n), replace=False)
    x[same_indices, 1] = x[same_indices, 0]
    
    # For other 50%, generate random floats
    diff_indices = np.setdiff1d(np.arange(n), same_indices)
    x[diff_indices, 1] = np.random.random(len(diff_indices))
    
    # Label 1 if same, 0 if different
    labels = (x[:, 0] == x[:, 1]).astype(int)
    
    percentage_ones = np.sum(labels) / n
    print(f"percentage of 1s: {percentage_ones}")
    
    # Calculate baseline accuracy for random guessing
    # Random guessing accuracy = max(p, 1-p) where p is the proportion of positive class
    baseline_accuracy = max(percentage_ones, 1 - percentage_ones)
    print(f"baseline accuracy (random guessing): {baseline_accuracy:.3f}")
    
    return zip(x, labels)


def train(data, model, single_sample=False, verbose=False):
    correct_samples = 0.0
    total_samples = 0
    for x, label in data:
        total_samples += 1
        if verbose:
            model.print_activation()
        y = model.forward(x)
        if verbose:
            model.print_activation()
        for i in range(2):  
            model.forward()
            model.update()
            # model.print_activation()
            # print(f"y {y}")
        print(f"y {float(y[0]):.2f}")
        y = model.forward()
        model.update()
        
        for i in range(5):  
            model.forward(np.zeros(2))
        
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
    data = gen_data(n=20)

    model = SparseNN(
        input_dim=2, activation_fn=tanh, n_neurons=6
    )  # Change to input_dim=2
    model.print_weight_ratio()
    train(data, model, single_sample=False)
    model.print_weight_ratio()
    model.print_average_weight()


if __name__ == "__main__":
    main()
