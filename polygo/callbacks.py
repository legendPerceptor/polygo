"""
Callback Metrics System
Author: Brian Liu
Date: 2019/07/25
"""

import numpy as np
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report

class F1Score(Callback):
    """Evaluation system
        
    """
    def __init__(self, seq, preprocessor=None):
        super(F1Score, self).__init__()
        self.seq = seq
        self.p = preprocessor

    def get_lengths(self, y_true):
        lengths = []
        for y in np.argmax(y_true, -1):
            try:
                i = list(y).index(0) #???
            except ValueError:
                i = len(y)
            lengths.append(i)

        return lengths

    def on_epoch_end(self, epoch, logs={}):
        label_true = []
        label_pred = []
        for i in range(len(self.seq)):
            x_true, y_true = self.seq[i]
            lengths = self.get_lengths(y_true)
            y_pred = self.model.predict_on_batch(x_true)

            y_true = self.p.inverse_transform(y_true, lengths)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            # y_true and y_pred is iterable object, cannot use append
            label_true.extend(y_true)
            label_pred.extend(y_pred)
        
        score = f1_score(label_true, label_pred)
        print(f' - f1: {(score*100):04.2f}')
        print(classification_report(label_true, label_pred))
        logs['f1'] = score