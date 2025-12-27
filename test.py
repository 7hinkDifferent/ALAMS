from utils import get_dataset

dataset = get_dataset("CIFAR10")
_, train_data = dataset.get_train_data()
print(len(train_data))  # Print the number of training samples
val_data = dataset.get_val_data()
print(len(val_data))  # Print the number of validation samples
test_data = dataset.get_test_data()
print(len(test_data))  # Print the number of test samples