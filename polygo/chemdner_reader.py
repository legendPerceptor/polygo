from nltk.tokenize import punkt
from nltk.tokenize import word_tokenize
import numpy as np
import itertools
import pickle
import os

class ChemdnerReader(object):
    """The easist way to use is to
        reader = ChemdnerReader() 
        reader.read_data()
        reader.save()
    """

    def __init__(self, train_text, train_label, dev_text, dev_label, eval_text, eval_label):
        self._train_text_path = train_text
        self._train_label_path = train_label
        self._dev_text_path = dev_text
        self._dev_label_path = dev_label
        self._eval_text_path = eval_text
        self._eval_label_path = eval_label
        self.all_texts = ''
    
    def read_abstracts(self, file: str) -> dict:
        """Store the whole abstracts file into all_texts
            1. Article Identifier
            2. Title of the article
            3. Abstract of the article
        Args:
            file: the path to the abstracts file
        Returns:
            a dict object, article id as the key, a tuple of title and abstract as the value.
        """
        with open(file, 'r') as f:
            ret = dict()
            for line in f:
                if line[-1] == '\n':
                    line = line[:-1]  # removing the EOL character
                line_list = line.split('\t')
                assert len(line_list) == 3, f"ERROR1: This line dose not have 3 columns:\nFILE: {file}\n{line_list}"
                # line_list[1] is title, line_list[2] is abstract
                # line_list[0] is the id
                self.all_texts = self.all_texts + line_list[1] + ' ' + line_list[2] + ' '
                ret[line_list[0]] = line_list
        return ret
    
    def read_annotations(self, file: str) -> dict:
        """Read the annotation to memory
            1. Article identifier
            2. Type
            3. Start Offest
            4. End Offest
            5. Text String of the entity
            6. Type of chemical entity mention 
        Args:
            file: the path to the annotation file
        Returns:
            a dict object, article id as the key, a tuple of the other five columns as the value.
        """
        with open(file, 'r') as f:
            ret = dict()
            for line in f:
                if line[-1] == '\n':
                    line = line[:-1]  # removing EOL
                line_list = line.split('\t')
                assert len(line_list) == 6, f"ERROR2: This line dose not have 6 columns:\n{line_list}"
                line_list[2] = int(line_list[2])
                line_list[3] = int(line_list[3])
                if line_list[0] not in ret:
                    ret[line_list[0]] = {'T': list(), 'A': list()}
                ret[line_list[0]][line_list[1]].append(line_list)
        return ret

    def read_all(self):
        """A wrapper to read all abstracts and annotations
        """
        self.train_txt = self.read_abstracts(self._train_text_path)
        self.dev_txt = self.read_abstracts(self._dev_text_path)
        self.eval_txt = self.read_abstracts(self._eval_text_path)
        print(f"Finished reading abstracts.\n# of sentences read: Train: {len(self.train_txt)}, Dev: {len(self.dev_txt)}, Eval: {len(self.eval_txt)}")
        
        self.train_anno = self.read_annotations(self._train_label_path)
        self.dev_anno = self.read_annotations(self._dev_label_path)
        self.eval_anno = self.read_annotations(self._eval_label_path)
        print(f"Finished reading annotations")

        self.punkt_tokenizer = punkt.PunktSentenceTokenizer(self.all_texts)

    def _generate_labels(self, sentence: str, anno_list:list) -> list:
        """Generate BIO labels, private
        """
        anno_list.sort(key = lambda x:x[2])
        last_pos = 0
        sentence_lst = list()
        label_lst = list()
        for item in anno_list:
            start_pos = item[2]
            end_pos = item[3]
            part = word_tokenize(sentence[last_pos:start_pos])
            sentence_lst.extend(part)
            label_lst.extend([('O', '')] * len(part))
            part = word_tokenize(sentence[start_pos:end_pos])
            sentence_lst.extend(part)
            label_lst.extend([('B', item[5])] + [('I', item[5])] * (len(part)-1))
            last_pos = end_pos
        part = word_tokenize(sentence[last_pos:])
        sentence_lst.extend(part)
        label_lst.extend([('O', '')] * len(part))
        assert len(sentence_lst) == len(label_lst), f"ERROR3: Label and tokenized sentence length mismatch!\n" \
                        f"{sentence}\n{list(itertools.zip_longest(sentence_lst, label_lst))}\n{anno_list}"
        return (sentence_lst, label_lst)
        
    def get_labels(self, text: dict, annotations: dict):
        """Return a list of sentences and a list of labels
        For example, the list of labels look like
        [('B', 'ABBREVIATION'), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', ''), ('O', '')]
        Args:
            text: dict, article id as the key, a tuple as the value
            annotations: dict, article id as the key, a tuple as the value
        Returns:
            A list of sentence and a list of labels
        """
        sentence_lst = list()
        label_lst = list()
        for pmid in annotations:
            dct = annotations[pmid]
            if dct['T']:
                sentence = text[pmid][1]
                anno_list = dct['T']
                lst1, lst2 = self._generate_labels(sentence, anno_list)
                sentence_lst.append(lst1)
                label_lst.append(lst2)
            if dct['A']:
                full_abstract = text[pmid][2]
                sentences = self.punkt_tokenizer.tokenize(full_abstract)
                anno_list = dct['A']
                anno_list.sort(key=lambda x:x[2])
                for sentence in sentences:
                    sentence_offset_in_abstract = full_abstract.find(sentence)
                    anno_list_for_this_sentence = list()
                    if anno_list:
                        next_start_pos = anno_list[0][2]                    
                        while next_start_pos < sentence_offset_in_abstract + len(sentence):
                            tmp_anno = anno_list[0]
                            del anno_list[0]
                            tmp_anno[2] = tmp_anno[2] - sentence_offset_in_abstract
                            tmp_anno[3] = tmp_anno[3] - sentence_offset_in_abstract
                            anno_list_for_this_sentence.append(tmp_anno)
                            if anno_list:
                                next_start_pos = anno_list[0][2]
                            else:
                                break
                    lst1, lst2 = self._generate_labels(sentence, anno_list_for_this_sentence)
                    sentence_lst.append(lst1)
                    label_lst.append(lst2)
        return (sentence_lst, label_lst)


    def padding_2D(self, pylist: list, max_len: int, padding_val):
        for row in pylist:
            if len(row) < max_len:
                row += [padding_val for _ in range(max_len - len(row))]
            else:
                row = row[:max_len]
    
    def get_text_and_labels(self):
        """A wrapper to get text and labels
        Must read_all first to get files into the memory
        """
        self.train_tokenized_txt, self.train_label = self.get_labels(self.train_txt, self.train_anno)
        self.dev_tokenized_txt, self.dev_label = self.get_labels(self.dev_txt, self.dev_anno)
        #self.eval_tokenized_txt, self.eval_label = self.get_labels(self.eval_txt, self.eval_anno)

        self.train_lengths = np.array([len(x) for x in self.train_tokenized_txt])
        self.dev_lengths = np.array([len(x) for x in self.dev_tokenized_txt])
        #self.eval_lengths = np.array([len(x) for x in self.eval_tokenized_txt])

        # self.max_len = 75 # not good option, should be customizable
        # padding_2D(self.train_tokenized_txt, max_len, '')
        # padding_2D(self.dev_tokenized_txt, max_len, '')
        # padding_2D(self.eval_tokenized_txt, max_len, '')
        # padding_2D(self.train_label, max_len, ('P', ''))
        # padding_2D(self.dev_label, max_len, ('P', ''))
        # padding_2D(self.eval_label, max_len, ('P', ''))

    
    def save(self, path='.'):
        save_to_pickle(self.train_tokenized_txt, os.path.join(path,'x_train.pickle'))
        save_to_pickle(self.dev_tokenized_txt, os.path.join(path,'x_dev.pickle'))
        #save_to_pickle(self.eval_tokenized_txt, os.path.join(path,'x_eval.pickle'))
        save_to_pickle(self.train_label, os.path.join(path,'y_train.pickle'))
        save_to_pickle(self.dev_label, os.path.join(path,'y_dev.pickle'))
        #save_to_pickle(self.eval_label, os.path.join(path,'y_eval.pickle'))

    def read_data(self):
        """A wrapper to read the data
        """
        self.read_all()
        self.get_text_and_labels()
    
    def load(self, path='.'):
        pass

def save_to_pickle(obj, file):
    with open(file,'wb') as f:
        pickle.dump(obj,f)