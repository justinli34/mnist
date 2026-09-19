from mnist.network import Network
from mnist.util import load_data, score


def main():
    X_train, y_train = load_data(train=True)
    X_test, y_test = load_data(train=False)

    network = Network(input_dimension=784, output_dimension=10, depth=2, width=256)
    network.learn(X_train, y_train, learning_rate=0.01, num_epochs=10, mini_batch_size=64)

    print(score(network, X_test, y_test))
