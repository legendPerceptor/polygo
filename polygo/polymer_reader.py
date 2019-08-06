# Read the data
# The result are a list of tokenized sentences
# and a list of labels grouped by sentences
import csv
import numpy as np
import matplotlib.pyplot as plt
class PolymerReader(object):

    def __init__(self, data_path):
        self._data_path = data_path
    
    def read(self):
        """Read Polymer data
        
        Returns:
            train_x: list of tokenized sentences as train data
            train_y: list of training labels
            valid_x: list of tokenized sentences as validation data
            valid_y: list of 
            test_x:
            test_y:

        """
        if self._data_path is None:
            raise FileNotFoundError("Please provide data path")
        sentences = list() # tokenized sentences
        labels = list()
        token_lengths = list()
        DATASET_PATH = self._data_path
        with open(DATASET_PATH + '/training_corpus_labeled_pos.csv', 'r') as f:
            reader = csv.reader(f, delimiter='|', quotechar='"')
            lst = list(reader)
            train_num = len(lst)
            for index,item in enumerate(lst):
                try:
                    labels.append(eval(item[3]))
                    tmp_token_list = eval(item[2])
                    token_lengths.append(len(tmp_token_list))
                    sentences.append(tmp_token_list)
                    
                except:
                    print("[training_corpus_labeled_pos.csv] Line", index ,"goes wrong")
                    train_num = train_num - 1
                    continue
                
        with open(DATASET_PATH + '/val_corpus_labeled_pos.csv', 'r') as f:
            reader = csv.reader(f, delimiter='|', quotechar='"')
            lst = list(reader)
            val_num = len(lst)
            for index,item in enumerate(lst):
                try:
                    labels.append(eval(item[3]))
                    tmp_token_list = eval(item[2])
                    token_lengths.append(len(tmp_token_list))
                    sentences.append(tmp_token_list)
                except:
                    print("[val_corpus_labeled_pos] Line", index ,"goes wrong")
                    val_num = val_num - 1
                    continue
                
        with open(DATASET_PATH + '/test_corpus_labeled_pos.csv', 'r') as f:
            reader = csv.reader(f, delimiter='|', quotechar='"')
            lst = list(reader)
            test_num = len(lst)
            for index,item in enumerate(lst):
                try:
                    labels.append(eval(item[3]))
                    tmp_token_list = eval(item[2])
                    token_lengths.append(len(tmp_token_list))
                    sentences.append(tmp_token_list)
                except:
                    print("[test_corpus_labeld_pos] Line", index ,"goes wrong")
                    test_num = test_num -1
                    continue
        
        train_x = sentences[:train_num]
        train_y = labels[:train_num]
        valid_x = sentences[train_num:val_num+train_num]
        valid_y = labels[train_num:val_num+train_num]
        test_x = sentences[-test_num:]
        test_y = labels[-test_num:]

        return (train_x,train_y,valid_x,valid_y,test_x,test_y)




