from typing import Tuple, List
from abc import ABC, abstractmethod
from collections import Counter
import sklearn.metrics

from structures import MaskedDataset

import random
import regex
import csv
import os

import json
import pandas
import tempfile
import subprocess
import re
from copy import copy


# from tqdm.notebook import tqdm
from tqdm.autonotebook import tqdm
tqdm.pandas()


class AbstractDataset(ABC):
    def __init__(self, *args, path: str, n_max : int = -1, shuffle=True, **kwargs):
        self.dataset: DataSet = None
        self.classes = set()
        self.n_max = n_max
        self.load(path)
        if shuffle:
          random.seed(9721)
          random.shuffle(self.dataset)
        _, _, labels = zip(*self.dataset)
        print(Counter(labels))
        super().__init__()

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def name(self):
        pass

    def getData(self, training_data_share = 0.9) -> Tuple[MaskedDataset, MaskedDataset]:
        all_data = self.dataset[:self.n_max]
        n_split = round(training_data_share * len(all_data))
        train_sentences = all_data[:n_split]
        test_sentences = all_data[n_split:]
        return MaskedDataset(train_sentences), MaskedDataset(test_sentences)

    @staticmethod
    def run_evaluation(
        predictions:pandas.DataFrame, # expects a dataframe of the form [id, true-class, predicted-class]
      ):
      report = sklearn.metrics.classification_report(
        predictions.iloc[:, 1], # y_true
        predictions.iloc[:, 2], # y_true
        output_dict = True,
        zero_division=0, # ignore warnings and set invalid values to zero
        labels=list(set(predictions.iloc[:, 1])) # consider only labels from the test set
      )
      f1 = report['macro avg']['f1-score'] # return macro f1 by default
      return f1, report

# OffensEval 2019
class OffensEvalData(AbstractDataset):
    def load(self, path: str):
        off_data: DataSet = []
        label2Idx = {'NOT' : 0, 'OFF' : 1}
        with open(os.path.join(path, "offenseval-training-v1.tsv"), 'r', encoding='UTF-8') as f:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                if i == self.n_max:
                    break
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                instance = (i, tweet, items[2])
                off_data.append(instance)
                self.classes.add(items[2])
        self.dataset = off_data

        self.testset = []
        with open(os.path.join(path, "labels-levela.csv"), 'r', encoding='UTF-8') as f, open(os.path.join(path, "testset-levela.tsv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(g)
            # read instances
            for i, line in enumerate(g):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = f.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (i, tweet, label[1])
                self.testset.append(instance)

    def getData(self, training_data_share = 0.9) -> Tuple[MaskedDataset, MaskedDataset]:
        train_sentences = self.dataset[:self.n_max]
        test_sentences = self.testset
        return MaskedDataset(train_sentences), MaskedDataset(test_sentences)
    def name(self):
        return "offenseval"

# OffensEval 2020
class OffensEvalData2020A(AbstractDataset):
    def load(self, path: str):

        # combine training and test data from OE2019 task to one training dataset
        # use it in 10-fold CV scenario
        # predict on OE2020 test data

        path19 = 'datasets/OffensEval19'
        off_data: DataSet = []
        label2Idx = {'NOT' : 0, 'OFF' : 1}
        with open(os.path.join(path19, "offenseval-training-v1.tsv"), 'r', encoding='UTF-8') as f:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                if i == self.n_max:
                    break
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                instance = (i, tweet, items[2])
                off_data.append(instance)
                self.classes.add(items[2])
        self.dataset = off_data

        testset19 = []
        with open(os.path.join(path19, "labels-levela.csv"), 'r', encoding='UTF-8') as f, open(os.path.join(path19, "testset-levela.tsv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(g)
            # read instances
            for i, line in enumerate(g):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = f.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (i, tweet, label[1])
                testset19.append(instance)
        self.dataset += testset19

        # load OE2020 test data
        self.testset = []
        with open(os.path.join(path, "test_a_tweets.tsv"), 'r', encoding='UTF-8') as f, open(os.path.join(path, "englishA-goldlabels.csv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = g.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (items[0], tweet, label[1])
                self.testset.append(instance)


    def getData(self, training_data_share = 0.9) -> Tuple[MaskedDataset, MaskedDataset]:
        train_sentences = self.dataset
        test_sentences = self.testset
        return MaskedDataset(train_sentences), MaskedDataset(test_sentences)
    def name(self):
        return "offenseval"



# OffensEval 2020
class OffensEvalData2020B(AbstractDataset):
    def load(self, path: str):

        # combine training and test data from OE2019 task to one training dataset
        # use it in 10-fold CV scenario
        # predict on OE2020 test data

        path19 = 'datasets/OffensEval19'
        off_data: DataSet = []
        label2Idx = {'TIN' : 0, 'UNT' : 1}
        with open(os.path.join(path19, "offenseval-training-v1.tsv"), 'r', encoding='UTF-8') as f:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                if i == self.n_max:
                    break
                items = line.split('\t')

                if items[3] == "NULL":
                    continue

                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                instance = (i, tweet, items[3])
                off_data.append(instance)
                self.classes.add(items[3])
        self.dataset = off_data

        testset19 = []
        with open(os.path.join(path19, "labels-levelb.csv"), 'r', encoding='UTF-8') as f, open(os.path.join(path19, "testset-levelb.tsv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(g)
            # read instances
            for i, line in enumerate(g):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = f.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (i, tweet, label[1])
                testset19.append(instance)
        self.dataset += testset19

        # load OE2020 test data
        self.testset = []
        with open(os.path.join(path, "test_b_tweets.tsv"), 'r', encoding='UTF-8') as f, open(os.path.join(path, "englishB-goldlabels.csv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = g.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (items[0], tweet, label[1])
                self.testset.append(instance)


    def getData(self, training_data_share = 0.9) -> Tuple[MaskedDataset, MaskedDataset]:
        train_sentences = self.dataset
        test_sentences = self.testset
        return MaskedDataset(train_sentences), MaskedDataset(test_sentences)
    def name(self):
        return "offenseval"


# OffensEval 2020
class OffensEvalData2020C(AbstractDataset):
    def load(self, path: str):

        # combine training and test data from OE2019 task to one training dataset
        # use it in 10-fold CV scenario
        # predict on OE2020 test data

        path19 = 'datasets/OffensEval19'
        off_data: DataSet = []
        label2Idx = {'IND' : 0, 'GRP' : 1, 'OTH': 2}
        with open(os.path.join(path19, "offenseval-training-v1.tsv"), 'r', encoding='UTF-8') as f:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                if i == self.n_max:
                    break
                items = line.split('\t')

                label = items[4].strip()

                if label == "NULL":
                    continue

                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()

                instance = (i, tweet, label)
                off_data.append(instance)
                self.classes.add(label)
        self.dataset = off_data

        testset19 = []
        with open(os.path.join(path19, "labels-levelc.csv"), 'r', encoding='UTF-8') as f, open(os.path.join(path19, "testset-levelc.tsv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(g)
            # read instances
            for i, line in enumerate(g):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = f.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (i, tweet, label[1])
                testset19.append(instance)
        self.dataset += testset19

        # load OE2020 test data
        self.testset = []
        with open(os.path.join(path, "test_c_tweets.tsv"), 'r', encoding='UTF-8') as f, open(os.path.join(path, "englishC-goldlabels.csv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = g.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (items[0], tweet, label[1])
                self.testset.append(instance)

    def getData(self, training_data_share = 0.9) -> Tuple[MaskedDataset, MaskedDataset]:
        train_sentences = self.dataset
        test_sentences = self.testset
        return MaskedDataset(train_sentences), MaskedDataset(test_sentences)
    def name(self):
        return "offenseval"

class DavidsonHateSpeechData(AbstractDataset):
    def load(self, path: str):
        # id2label = {"0": "HATE", "1" : "OFF", "2" : "NOT"}
        id2label = {"0": "OFF", "1" : "OFF", "2" : "NOT"}
        twitter_replacer = regex.compile(r'@([A-Za-z0-9_]+) ?|http[s]?://\S+ ?|&#\d+;')
        all_data = []
        with open(path, 'r', encoding='UTF-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            next(f)
            for row in reader:
                tweet = regex.sub(twitter_replacer, " ", row[6]).strip()
                tweet = regex.sub("^RT[: ]+", "", tweet)
                tweet = regex.sub("\s\s+", " ", tweet)
                if tweet:
                    label = id2label[row[5]]
#                         if row[5] == "2":
#                             label = 'NOT'
#                         elif row[5] == "0":
#                             label = 'OFF'
#                         else:
#                             continue
                    instance = (row[0], tweet, label)
                    all_data.append(instance)
                    self.classes.add(row[5])
        self.dataset = all_data
    def name(self):
        return "hatespeech"

