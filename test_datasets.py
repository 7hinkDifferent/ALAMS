from torchvision import datasets
from utils import get_dataset

def test_mnist():
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    assert raw_train.data.shape[0] == 60000
    assert raw_test.data.shape[0] == 10000
    print("MNIST dataset test passed.")

def test_fashion_mnist():
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    assert raw_train.data.shape[0] == 60000
    assert raw_test.data.shape[0] == 10000
    print("FashionMNIST dataset test passed.")

def test_svhn():
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    assert data_train.data.shape[0] == 73257
    assert data_test.data.shape[0] == 26032
    print("SVHN dataset test passed.")

def test_cifar10():
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    assert data_train.data.shape[0] == 50000
    assert data_test.data.shape[0] == 10000
    print("CIFAR10 dataset test passed.")

if __name__ == "__main__":
    # test_mnist()
    # test_fashion_mnist()
    # test_svhn()
    # test_cifar10()
    print(get_dataset("MNIST").get_statistics())
    print(get_dataset("FashionMNIST").get_statistics())
    print(get_dataset("SVHN").get_statistics())
    print(get_dataset("CIFAR10").get_statistics())