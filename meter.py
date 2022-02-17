import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys


class Meter(object):
    """
    The pre processing data to sutibale format for using skLearn metrics.
    change tensor data in pytorch to numpy format.
    """

    def __init__(self):
        """
        Args:
            classs: label of classes in the classification problem
        """
        self.classes = [str(i) for i in range(60)]
        self.predc_array = []
        self.label_array = []


    def add(self, predicted, target):
        """
        Add and convert params to sutibale params for making confusion matrix
        Args:
            predicted (tensor), target (tensor)
        Attention:
            If you have batch data just call add function in loop.
        """
        #convert tensor to numpy
        predc = predicted.cpu().squeeze().numpy()
        label = target.data.cpu().squeeze().numpy()
        #make one array to confusion matrix
        self.predc_array.extend(predc)
        self.label_array.extend(label)

    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm *= 100

        plt.figure(figsize=(50,50))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.axis('scaled')

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Actual class')
        plt.xlabel('Predicted class')
    
    def confusion_figure(self, normalize=False,
                          title='Confusion matrix',
                          saving_path = False,
                          cmap=plt.cm.Blues):
        cnf_matrix = confusion_matrix(self.label_array, self.predc_array)
        np.set_printoptions(precision=0)
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=self.classes,
                            normalize = normalize,
                            title = title,
                            cmap  = cmap)
        if (saving_path != False):    
            plt.savefig(saving_path)
        plt.cla()
        plt.close('all')
    def classification_metrics(self):
        print(classification_report(self.label_array, self.predc_array,
                                    target_names=self.classes))
        
                
    

        

