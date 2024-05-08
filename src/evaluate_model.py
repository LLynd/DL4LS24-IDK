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