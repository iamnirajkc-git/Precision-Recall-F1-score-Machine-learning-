# Precision-Recall-F1-score-Machine-learning-
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix , classification_report
truth = ["Dog","Not a dog","Dog","Dog", "Dog", "Not a dog", "Not a dog", "Dog", "Dog", "Not a dog"]
prediction = ["Dog","Dog", "Dog","Not a dog","Dog", "Not a dog", "Dog", "Not a dog", "Dog", "Dog"]
cm = confusion_matrix(truth,prediction)
import seaborn as sns
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):


    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names,)
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
plt.ylabel('Truth')
plt.xlabel('Prediction')
print_confusion_matrix(cm,["Dog","Not a dog"])
print(classification_report(truth, prediction))
"f1 score for Dog class :",2*(0.57*0.67/(0.57+0.67)) ,",f1 score for not Dog class",2*(0.33*0.25/(0.33+0.25))



