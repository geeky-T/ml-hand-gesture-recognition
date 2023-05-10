import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc, cohen_kappa_score
from sklearn import metrics


all_labels = [
    # 'back_left',
    'back',
    # 'blank',
    # 'buy_left',
    'buy',
    # 'more_left',
    'more',
    # 'next_left',
    'next',
    # 'previous_left',
    'previous'
    ]
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(all_labels): # all_labels: no of the labels, for ex. ['cat', 'dog', 'rat']
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average)
  
  
  # print confusion matrix and calculate accuracy rate
def print_performance(pred, actual):
    pred = pd.Series(pred).map(int)
    actual = pd.Series(actual).map(int)
    actual_array = np.array(actual)
    unique_label = np.unique([actual, pred])
    cf = pd.DataFrame(
        confusion_matrix(actual_array, pred, labels=unique_label),
        index=['Actual:{:}'.format(x) for x in unique_label],
        columns=['Pred:{:}'.format(x) for x in unique_label]
    )
    sns.heatmap(cf, annot=True, cmap='YlGnBu', fmt='.8g')
    plt.show()
    print(cf)
    print(metrics.classification_report(pred, actual, target_names=all_labels))
    print("Accuracy of the test set: ", accuracy_score(actual, pred))
    print("Cohen Kappa Value: ", cohen_kappa_score(actual, pred))
    print('Percent Back correctly predicted: ',
          cf['Pred:0'][0]/(cf['Pred:0'][0] + cf['Pred:1'][0] + cf['Pred:2'][0] + cf['Pred:3'][0] + cf['Pred:4'][0])*100)
    print('Percent Buy correctly predicted: ',
          cf['Pred:1'][1]/(cf['Pred:0'][1] + cf['Pred:1'][1] + cf['Pred:2'][1] + cf['Pred:3'][1] + cf['Pred:4'][1])*100)
    print('Percent More correctly predicted: ',
          cf['Pred:2'][2]/(cf['Pred:0'][2] + cf['Pred:1'][2] + cf['Pred:2'][2] + cf['Pred:3'][2] + cf['Pred:4'][2])*100)
    print('Percent Next correctly predicted: ',
          cf['Pred:3'][3]/(cf['Pred:0'][3] + cf['Pred:1'][3] + cf['Pred:2'][3] + cf['Pred:3'][3] + cf['Pred:4'][3])*100)
    print('Percent Previous correctly predicted: ',
          cf['Pred:4'][4]/(cf['Pred:0'][4] + cf['Pred:1'][4] + cf['Pred:2'][4] + cf['Pred:3'][4] + cf['Pred:4'][4])*100)