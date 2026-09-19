import numpy as np
import numpy.typing as npt
from torchvision.datasets import MNIST

from mnist import SEED
from mnist.network import Network


def load_data(*, train: bool) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Returns (X, y) for the requested split. X is flattened and normalized."""
    dataset = MNIST(root="./data", train=train, download=True)
    X = dataset.data.numpy().astype(np.float32)
    X = X.reshape(len(X), -1) / 255.0
    y = dataset.targets.numpy().astype(np.int64)
    return X, y


def split_data(
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.int64],
    split_ratio: float,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
    npt.NDArray[np.int64],
]:
    """Returns (X_first, y_first, X_second, y_second) after shuffling.

    split_ratio: fraction of examples in the second split.
    """
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X))
    split_idx = int(len(X) * (1 - split_ratio))
    first_idx = indices[:split_idx]
    second_idx = indices[split_idx:]
    X_first = X[first_idx]
    y_first = y[first_idx]
    X_second = X[second_idx]
    y_second = y[second_idx]
    return X_first, y_first, X_second, y_second


def score(
    network: Network,
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.int64] | np.int64 | int,
) -> float:
    """Returns the accuracy of the network on the given data."""
    y_pred = network.predict(X)
    accuracy = np.mean(y_pred == np.atleast_1d(y))
    return float(accuracy)
