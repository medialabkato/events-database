import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

class ColumnSelector(object):
    """
    A feature selector for scikit-learn's Pipeline class that returns
    specified columns from a numpy array.

    """

    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, y=None):
        return X[self.cols]

    def fit(self, X, y=None):
        return self


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,size=5):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    
    plt.figure(figsize = (size,size))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.savefig('/home/as/Pulpit/Medialab/fb/events/typ_wydarzenia')

class MakeDummies(object):
    """
    A feature selector for scikit-learn's Pipeline class that returns
    specified columns from a numpy array.

    """

    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, y=None):
        return pd.get_dummies(X, columns=self.cols)

    def fit(self, X, y=None):
        return self





def lemmatize_text(text):
    import json
    import requests
    from xml.etree import ElementTree as ET
    url = 'http://ws.clarin-pl.eu/nlprest2/base'
    data = {'lpmn':'any2txt|maca({"morfeusz2":true})', 'text': text, 'user' : 'moj@adres.mail'}
    r = requests.post(url + '/process', data=json.dumps(data))
    tree = ET.fromstring(r.text)
    return ' '.join([tok.find('.//base').text.split(':')[0] for tok in tree.findall('.//tok')])



def lemmatize(text):
    import subprocess
    cnt[0] = cnt[0] + 1
    if cnt[0] % 50 == 0:
        print(cnt)
    process = subprocess.Popen('python2.7 lemmatizer.py \'' + text + '\'', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output = process.communicate()[0]
    return output.decode('utf-8')
    

