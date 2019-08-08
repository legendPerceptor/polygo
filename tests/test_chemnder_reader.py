import os
import shutil
import unittest

from polygo.chemdner_reader import ChemdnerReader

class TestChemdnerReader(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.train_text = os.path.join(os.path.dirname(__file__), "../data/chemdner_corpus/training.abstracts.txt")
        cls.train_label = os.path.join(os.path.dirname(__file__), "../data/chemdner_corpus/training.annotations.txt")
        cls.dev_text = os.path.join(os.path.dirname(__file__), "../data/chemdner_corpus/development.abstracts.txt")
        cls.dev_label = os.path.join(os.path.dirname(__file__), "../data/chemdner_corpus/development.annotations.txt")
        cls.eval_text = os.path.join(os.path.dirname(__file__), "../data/chemdner_corpus/evaluation.abstracts.txt")
        cls.eval_label = os.path.join(os.path.dirname(__file__), "../data/chemdner_corpus/evaluation.annotations.txt")
        cls.save_dir = os.path.join(os.path.dirname(__file__), "chemdner_data")
        if not os.path.exists(cls.save_dir):
            os.mkdir(cls.save_dir)
    
    def testReadAndSave(self):
        reader = ChemdnerReader(self.train_text,self.train_label,
        self.dev_text,self.dev_label,self.eval_text,self.eval_label)
        reader.read_data()
        print(reader.train_tokenized_txt[5])
        print(reader.train_label[5])
        reader.save(self.save_dir)