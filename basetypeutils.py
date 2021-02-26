#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:06:10 2020

@author: remstef
"""

from typing import List, Tuple, Dict
from torch import tensor
from flair.data import Token as FlairToken, Sentence as FlairSentence

# ids, texts, labels
DataSet = List[Tuple[str, str, str]]
# sim, test.label, train.label, test.text, train.text, test.context, train.context, test.id, train.id
StructurePairResult = Tuple[float, str, str, str, str, str, str, str]
ResultSet = List[StructurePairResult]
# confidence, predicted label, true label, text, most_sim_train_text
RetrievalResult = Tuple[float, str, str, str, str]

# classes
# -------
class MaskableToken(FlairToken):
    def __init__(self, *args, **kwargs):
        super(MaskableToken, self).__init__(*args, **kwargs)
        self._init_MaskableToken()
    def _init_MaskableToken(self):
        self.mask_id: int = 0
        self.sum_embedding: tensor = None
    def setMask(self, id: int):
        self.mask_id = id
    def getMask(self):
        return self.mask_id


class MaskableSentence(FlairSentence):
    def __init__(self, *args, sid=None, **kwargs):
        super(MaskableSentence, self).__init__(*args, **kwargs)
        self.properties: Dict = { 'sid', sid } if sid is not None else {}

# complex datatypes
# -----------------
class MaskedStructure(object):
    def __init__(self, text: str, context: str, embedding: tensor, label: str = None, id = None):
        self.id = id
        self.text: str = text
        self.context: str = context
        self.label: str = label
        self.embedding: tensor = embedding
    def __repr__(self):
        max_len_print = 50
        if len(self.text) > max_len_print:
            text = self.text[:(max_len_print - 3)] + "..."
        else:
            text = self.text.strip()
        return str((self.id, self.label, text))

# types
LabeledData = Dict[str, List[MaskedStructure]]

# OTHER
# -----

class AttributeHolder(object):
  def __init__(self, **kwargs):
    [ self.__setitem__(k,v) for k,v in kwargs.items() ]
  def __repr__(self):
    return f'{self.__class__.__name__:s}({self.__dict__.__repr__():s})'
  def __setitem__(self, key, value):
    return setattr(self, key, value)
  def __getitem__(self, key):
    if not hasattr(self, key):
      return None
    return getattr(self, key)
  # def dump(self, dest=None):
  #   if isinstance(dest, str):
  #     with open(dest, 'wt') as f:
  #       return yaml.dump(self.__dict__, f)
  #   return yaml.dump(self.__dict__, dest)
  # def load(self, src, keep=[]):
  #   if isinstance(src, str):
  #     with open(src, 'rt') as f:
  #       d = yaml.load(f, Loader=yaml.FullLoader)
  #   else:
  #     d = yaml.load(src, Loader=yaml.FullLoader)
  #   [ self.__setitem__(k,v) for k,v in d.items() if k not in keep ]
  #   return self
  def has(self, key):
    if hasattr(self, key):
      return self.__getitem__(key) is not None
    return False