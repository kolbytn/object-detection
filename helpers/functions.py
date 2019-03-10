import matplotlib.pyplot as plt
from data import ObjectDataset


def plot_results(data):
    plt.plot(data)
    plt.show()


def load_data(directory_path):
    # Get raw data from path
    data = None

    # Process data
    training_dataset = None
    testing_dataset = None

    # Create Dataset
    training_dataset = ObjectDataset(training_features, training_labels)
    testing_dataset = ObjectDataset(testing_features, testing_labels)

    return training_dataset, testing_dataset
