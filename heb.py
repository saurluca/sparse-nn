import numpy as np


def sigmoid(x):
    """Sigmoid activation function"""
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))


def tanh_activation(x):
    """Tanh activation function"""
    return np.tanh(x)


def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)


class HebbianSparseNN:
    """
    Sparse Neural Network with Hebbian Learning inspired by research.
    Combines unsupervised Hebbian feature learning with supervised classification.
    """

    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Hebbian learning parameters
        self.hebb_lr = 0.05
        self.class_lr = 0.1
        self.activation_threshold = 0.1  # Threshold for neuron "firing"
        self.weight_decay = 0.995

        # Network structure: input -> hidden (Hebbian) -> output (supervised)
        self.activations = {
            "input": np.zeros(input_dim),
            "hidden": np.zeros(hidden_dim),
            "output": np.zeros(output_dim),
        }

        self.prev_activations = {
            "input": np.zeros(input_dim),
            "hidden": np.zeros(hidden_dim),
            "output": np.zeros(output_dim),
        }

        # Initialize sparse connectivity
        self._initialize_weights()

        print(f"HebbianSparseNN: {input_dim} -> {hidden_dim} -> {output_dim}")
        print(f"Hebbian connections: {np.sum(self.hebb_mask)}")
        print(f"Classification connections: {np.sum(self.class_mask)}")

    def _initialize_weights(self):
        """Initialize sparse weight matrices"""
        # Hebbian layer: input -> hidden (sparse connections)
        self.hebb_weights = np.zeros((self.hidden_dim, self.input_dim))
        self.hebb_mask = np.zeros((self.hidden_dim, self.input_dim), dtype=bool)

        # Create sparse connections (70% connectivity)
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
                if np.random.random() > 0.3:
                    self.hebb_mask[i, j] = True
                    self.hebb_weights[i, j] = np.random.normal(0, 0.3)

        # Classification layer: hidden -> output (dense connections)
        self.class_weights = np.zeros((self.output_dim, self.hidden_dim))
        self.class_mask = np.ones((self.output_dim, self.hidden_dim), dtype=bool)
        self.class_weights = np.random.normal(
            0, 0.1, (self.output_dim, self.hidden_dim)
        )

        # Bias terms
        self.hidden_bias = np.random.normal(0, 0.1, self.hidden_dim)
        self.output_bias = np.random.normal(0, 0.1, self.output_dim)

    def forward(self, x=None):
        """Forward pass through the network"""
        # Store previous activations
        for key in self.activations:
            self.prev_activations[key] = self.activations[key].copy()

        # Set input
        if x is not None:
            self.activations["input"] = x.copy()

        # Hidden layer (Hebbian learned features)
        hidden_input = np.zeros(self.hidden_dim)
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
                if self.hebb_mask[i, j]:
                    hidden_input[i] += (
                        self.hebb_weights[i, j] * self.activations["input"][j]
                    )
            hidden_input[i] += self.hidden_bias[i]

        self.activations["hidden"] = tanh_activation(hidden_input)

        # Output layer (supervised classification)
        output_input = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            for j in range(self.hidden_dim):
                output_input[i] += (
                    self.class_weights[i, j] * self.activations["hidden"][j]
                )
            output_input[i] += self.output_bias[i]

        self.activations["output"] = sigmoid(output_input)

        return self.activations["output"]

    def hebbian_update(self):
        """
        Unsupervised Hebbian learning for feature extraction.
        Based on "neurons that fire together, wire together"
        """
        # Update Hebbian weights (input -> hidden)
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
                if self.hebb_mask[i, j]:
                    pre = self.activations["input"][j]
                    post = self.activations["hidden"][i]

                    # Activation threshold: only update if neuron is "firing"
                    if abs(post) > self.activation_threshold:
                        # Standard Hebbian rule with normalization
                        delta_w = self.hebb_lr * pre * post

                        # Weight normalization (Oja's rule component)
                        weight_norm = np.sum(self.hebb_weights[i, :] ** 2) + 1e-8
                        delta_w -= (
                            self.hebb_lr
                            * post
                            * post
                            * self.hebb_weights[i, j]
                            / weight_norm
                        )

                        self.hebb_weights[i, j] += delta_w

                        # Apply weight decay
                        self.hebb_weights[i, j] *= self.weight_decay

                        # Clip weights
                        self.hebb_weights[i, j] = np.clip(
                            self.hebb_weights[i, j], -2, 2
                        )

    def supervised_update(self, target):
        """
        Supervised learning for classification layer.
        Simple gradient descent on output error.
        """
        # Compute output error
        output_error = target - self.activations["output"]

        # Update output weights and bias
        for i in range(self.output_dim):
            # Update weights
            for j in range(self.hidden_dim):
                delta_w = (
                    self.class_lr * output_error[i] * self.activations["hidden"][j]
                )
                self.class_weights[i, j] += delta_w

            # Update bias
            self.output_bias[i] += self.class_lr * output_error[i]

        # Clip classification weights
        self.class_weights = np.clip(self.class_weights, -3, 3)
        self.output_bias = np.clip(self.output_bias, -2, 2)

    def print_stats(self):
        """Print network statistics"""
        hebb_weights_connected = self.hebb_weights[self.hebb_mask]
        pos_hebb = np.sum(hebb_weights_connected > 0)
        neg_hebb = np.sum(hebb_weights_connected < 0)
        avg_hebb = (
            np.mean(hebb_weights_connected) if len(hebb_weights_connected) > 0 else 0
        )

        avg_class = np.mean(self.class_weights)

        print(
            f"Hebbian weights - Pos: {pos_hebb}, Neg: {neg_hebb}, Avg: {avg_hebb:.4f}"
        )
        print(f"Classification weights - Avg: {avg_class:.4f}")
        print(f"Output: {self.activations['output']}")


def generate_comparison_data(n_samples=100, noise_level=0.05):
    """Generate comparison task data with some noise"""
    X = np.random.uniform(0.2, 0.8, (n_samples, 2))

    # Add small amount of noise to make task more realistic
    X += np.random.normal(0, noise_level, X.shape)
    X = np.clip(X, 0, 1)

    y = (X[:, 0] > X[:, 1]).astype(float)

    print(f"Generated {n_samples} samples")
    print(f"Positive ratio: {np.mean(y):.3f}")

    return X, y


def train_hebbian_network(model, X, y, epochs=200, unsupervised_steps=3):
    """
    Train network with two phases:
    1. Unsupervised Hebbian learning for feature extraction
    2. Supervised learning for classification
    """
    n_samples = len(X)

    for epoch in range(epochs):
        correct = 0
        indices = np.random.permutation(n_samples)

        for idx in indices:
            x_sample = X[idx]
            y_sample = y[idx]

            # Phase 1: Unsupervised feature learning
            model.forward(x_sample)

            # Multiple steps of unsupervised Hebbian learning
            for _ in range(unsupervised_steps):
                model.hebbian_update()
                model.forward()  # Update activations

            # Phase 2: Supervised classification learning
            model.supervised_update(np.array([y_sample]))

            # Final prediction
            output = model.forward()
            prediction = output[0] > 0.5

            if prediction == (y_sample > 0.5):
                correct += 1

        accuracy = correct / n_samples

        if epoch % 40 == 0:
            print(f"Epoch {epoch}: Accuracy = {accuracy:.3f}")

    return accuracy


def main():
    """Main function demonstrating Hebbian learning"""
    print("=== Hebbian Sparse Neural Network ===")
    print("Inspired by 'neurons that fire together, wire together'")
    print("Task: Which number is bigger?\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate data
    X_train, y_train = generate_comparison_data(100, noise_level=0.03)
    X_test, y_test = generate_comparison_data(50, noise_level=0.03)

    # Create network
    model = HebbianSparseNN(input_dim=2, hidden_dim=8, output_dim=1)

    print("\nInitial state:")
    model.print_stats()

    # Train with Hebbian learning
    print("\nTraining with Hebbian learning...")
    train_hebbian_network(model, X_train, y_train, epochs=200)

    print("\nFinal state:")
    model.print_stats()

    # Test performance
    print("\nTesting...")
    test_correct = 0
    test_outputs = []

    for i in range(len(X_test)):
        # Reset network state
        for key in model.activations:
            model.activations[key] = np.zeros_like(model.activations[key])

        output = model.forward(X_test[i])
        prediction = output[0] > 0.5
        true_label = y_test[i] > 0.5

        if prediction == true_label:
            test_correct += 1

        test_outputs.append(output[0])

    test_accuracy = test_correct / len(X_test)
    print(f"Test accuracy: {test_accuracy:.3f}")
    print("Random baseline: 0.500")
    print(f"Output range: [{min(test_outputs):.3f}, {max(test_outputs):.3f}]")

    # Show detailed examples
    print("\nDetailed examples:")
    for i in range(min(10, len(X_test))):
        # Reset state
        for key in model.activations:
            model.activations[key] = np.zeros_like(model.activations[key])

        output = model.forward(X_test[i])
        pred = output[0] > 0.5
        true_val = y_test[i] > 0.5
        diff = abs(X_test[i][0] - X_test[i][1])

        correct_mark = "✓" if pred == true_val else "✗"
        confidence = abs(output[0] - 0.5) * 2  # Convert to 0-1 scale

        print(
            f"[{X_test[i][0]:.3f}, {X_test[i][1]:.3f}] (diff: {diff:.3f}) -> "
            f"{output[0]:.3f} ({confidence:.3f}) -> {pred} (true: {true_val}) {correct_mark}"
        )

    return model


if __name__ == "__main__":
    trained_model = main()
