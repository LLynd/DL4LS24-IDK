import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_uncertainties(uncertainties):
    plt.hist(uncertainties, bins=20)
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Distribution of Uncertainties')
    plt.show()

def bar_plot_accuracy_per_class(preds, labels):
    '''
    Input: numpy arrays of model predictions and labels
    '''

    def one_hot_encoding(array, num_classes = 14):
        tensor = torch.tensor(array)
        one_hot = torch.zeros(tensor.size(0), num_classes)
        one_hot.scatter_(1, tensor.unsqueeze(1).long(), 1)
        return one_hot.numpy()

    def calculate_accuracy_across_clases(preds, labels):
        pred_one_hot = one_hot_encoding(preds)
        labels_one_hot = one_hot_encoding(labels)
        corect = pred_one_hot * labels_one_hot
        total_number_of_examples_per_class = np.sum(labels_one_hot, axis = 0)
        total_number_of_corect_examples_per_class = np.sum(corect, axis = 0)
        return total_number_of_corect_examples_per_class/ total_number_of_examples_per_class
    
    acuracy_per_class = calculate_accuracy_across_clases(preds, labels)

    list_of_class_labels = ['MacCD163', 'Mural', 'DC', 'Tumor', 'CD4', 'HLADR', 'NK', 'CD8', 'Treg', 'Neutrophil', 'plasma', 'B', 'pDC', 'BnT']

    plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 4000, len(labels)))

    # Create a bar plot with rainbow colors

    bars = plt.bar(np.arange(len(acuracy_per_class)), acuracy_per_class, color=colors)

    plt.xticks(np.arange(len(acuracy_per_class)), list_of_class_labels)

    i = 0
    for bar in bars:
        height = acuracy_per_class[i]
        plt.text(bar.get_x() + bar.get_width() / 2, height, round(height, 2), ha='center', va='bottom')
        # plt.text(bar.get_x() + bar.get_width() / 2, height, '%d' % height, ha='center', va='bottom')
        i = i+1

    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per class')
    plt.show()