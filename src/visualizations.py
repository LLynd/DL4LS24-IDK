import numpy as np
import matplotlib.pyplot as plt


def visualize_uncertainties(uncertainties):
    plt.hist(uncertainties, bins=20)
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Distribution of Uncertainties')
    plt.show()