import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def evaluate_model(y_predicted, y_predicted_proba, y_true):
    accuracy = accuracy_score(y_true, y_predicted)
    print("Accuracy:", accuracy)

    macro_f1 = f1_score(y_true, y_predicted, average='macro')
    print("Macro F1 score:", macro_f1)

    auc = roc_auc_score(y_true, y_predicted_proba, multi_class='ovr')
    print("AUC score (OvR):", auc)

    average_precision = average_precision_score(y_true, y_predicted_proba)
    print("Average Precision score:", average_precision)
    
    return accuracy, macro_f1, auc, average_precision

def one_hot_encoding(array, num_classes = 14):
    # Create a zero tensor with the desired shape
    tensor = torch.tensor(array)
    one_hot = torch.zeros(tensor.size(0), num_classes)
    # Use scatter_ to fill the one-hot tensor
    one_hot.scatter_(1, tensor.unsqueeze(1).long(), 1)
    return one_hot.numpy()

def calculate_accuracy_across_clases(model, x_test, y_test):
    y_pred = model.predict(x_test)
    pred_one_hot = one_hot_encoding(y_pred)
    labels_one_hot = one_hot_encoding(y_test)
    corect = pred_one_hot * labels_one_hot
    total_number_of_examples_per_class = np.sum(labels_one_hot, axis = 0)
    total_number_of_corect_examples_per_class = np.sum(corect, axis = 0)
    print(f'Total no of examples:, total no of corect ansvers:, accuracy per class: \n {total_number_of_examples_per_class} \n {total_number_of_corect_examples_per_class} \n {np.round(total_number_of_corect_examples_per_class /total_number_of_examples_per_class, 4)}')
