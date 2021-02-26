#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:54:26 2020

@author: remstef
"""

from typing import List, Tuple, Callable
from spacy.tokens.doc import Doc as SpacyDoc
from spacy.tokens.token import Token as SpacyToken
from flair.data import Token as FlairToken
from basetypeutils import MaskableToken

from pandas.core.frame import DataFrame
from pandas.core.series import Series
import networkx as nx


def get_row(df:DataFrame, field: str, value: object) -> Tuple[int, Series]:
  sub = df[df[field] == value]
  if sub.shape[0] == 0:
    return (-1, None)
  if sub.shape[0] > 1:
    print(f'''Warning: more than one value found for query: '{value:s}' on field '{field:s}'  ! Returning only first!''')
  return (sub.index[0], sub.iloc[0])


def get_spacy_doc_for_text(df:DataFrame, text:str) -> SpacyDoc:
  i, r = get_row(df, 'sentence', text)
  return None if i < 0 else r.spacydoc


def get_spacy_doc_for_id(df:DataFrame, strid:str) -> SpacyDoc:
  i, r = get_row(df, 'strid', strid)
  return None if i < 0 else r.spacydoc


def build_pandas_id_tokenizer(some_id_or_text:str, getterfunction: Callable[[str], SpacyDoc]) -> Callable[[str], List[FlairToken]]:

  doc: SpacyDoc = getterfunction(some_id_or_text)

  def get_tokens__build_pandas_id_tokenizer(some_irrelevant_text_which_will_be_ignored: str):
    previous_token = None
    tokens: List[FlairToken] = []
    for word in doc:
      word: SpacyToken = word
      token = FlairToken(
        text=word.text, start_position=word.idx, whitespace_after=True
      )
      tokens.append(token)

      if (previous_token is not None) and (
        token.start_pos - 1
        == previous_token.start_pos + len(previous_token.text)
      ):
        previous_token.whitespace_after = False

      previous_token = token
    return tokens

  return get_tokens__build_pandas_id_tokenizer



def build_spacy_flair_adapter(getterfunction: Callable[[str], SpacyDoc]) -> Callable[[str], List[FlairToken]]:

  def get_tokens__build_spacy_flair_adapter(some_text: str):
    doc: SpacyDoc = getterfunction(some_text)
    previous_token = None
    tokens: List[FlairToken] = []
    for word in doc:
      word: SpacyToken = word
      token = FlairToken(
        text=word.text, start_position=word.idx, whitespace_after=True
      )
      tokens.append(token)

      if (previous_token is not None) and (
        token.start_pos - 1
        == previous_token.start_pos + len(previous_token.text)
      ):
        previous_token.whitespace_after = False

      previous_token = token
    return tokens

  return get_tokens__build_spacy_flair_adapter



def get_annotated_sentence(doc: SpacyDoc, pos=True, tag=True, dependencyhead=True, lemma=True, netype=True, ne=True, sentence_offset=0) -> str:
  '''
  Returns
  -------
  str
    Syntactic sentence in the form of:
      'word/POS/dependencyrelationname:dependencyrelationheadindex/lemma /entities...'
      head index is starting from 1, root relation has index 0
  '''
  ents = ' ' + ' '.join([
    '/'.join([
      f'/{e.start-sentence_offset+1}-{e.end-sentence_offset}/////',
      e.text.replace(' ','_'),
      e.label_,
      e.kb_id_]) for e in doc.ents])
  return (' '.join([
    '/'.join([
      t.text,
      t.pos_ if pos else '',
      f'''{t.dep_}:{(t.head.i-sentence_offset+1) if t.head.i != t.i else '0'}''' if dependencyhead else '',
      t.tag_ if tag else '',
      t.lemma_ if lemma else '',
      t.ent_type_ if netype else ''
    ]) for t in doc]) + ents if ne else '').strip()


def get_annotated_sentence_path_elements(doc: SpacyDoc, src, dest, path, sentence_offset=0) -> str:
  path_tokens = [t for t in path if isinstance(t, SpacyToken)] + [doc[i] for i in src] + [doc[i] for i in dest]
  def make_annotation(t):
    return f'{t.dep_}:{(t.head.i-sentence_offset+1) if t.head.i != t.i else 0}'
  annotated_sentence = ' '.join([
    '/'.join([
      t.text,
      '',
      f'e1/{make_annotation(t)}' if t.i in src else (f'e2/{make_annotation(t)}' if t.i in dest else (f'p/{make_annotation(t)}' if t in path_tokens else '')),
    ]) for t in doc])
  return annotated_sentence


def get_spacy_offsets(doc:SpacyDoc, charoffset:Tuple[int, int]) -> Tuple[int, int]:
  # offsets
  ob, oe = charoffset
  b = next((t.i for t in doc if ob < t.idx+len(t)), None)
  e = next((t.i for t in doc[b+1:] if t.idx >= oe), len(doc) if oe == len(doc.text) else None)
  return (b, e)


def get_shortest_dependency_path(doc: SpacyDoc, src_ids: List[int], dst_ids: List[int]) -> Tuple[List[object], List[object], List[Tuple[int, int]], object]:
  '''
    get the shortest dependency path from one of the nodeindices
    in source to one of the node indices in dest
  '''
  # debug:
  # - display dependencies
  #     from spacy import displacy
  #     displacy.serve(doc, style='dep')
  #     displacy.render(doc, style='dep')
  # - display graph
  #     import matplotlib.pyplot as plt
  #     nx.draw_shell(G, with_labels=True)

  # build graph
  G = nx.Graph()
  for t in doc:
    if t.head == t: # skip root node as it would introduce a cycle
      continue
    G.add_node(t.i, token=t)
    G.add_node(t.head.i, token=t.head)
    G.add_edge(t.head.i, t.i, relation=f'-{t.dep_:s}->', full_relation=f'{t.dep_:s}({t.head}, {t})', head=t.head.i, tail=t.i)
  # compute shortest path between any of the nodes in source and dest
  try:
    npath = nx.shortest_path(G, src_ids[0], dst_ids[-1])
    epath = list(nx.utils.pairwise(npath))
    dpath = [ [ G.edges[e]['relation'] if not ((e[0] in src_ids and e[1] in src_ids) or (e[0] in dst_ids and e[1] in dst_ids)) else None, G.nodes[e[1]]['token'] if not (e[1] in src_ids or e[1] in dst_ids) else None ] for e in epath ]
    dpath = [ i for pi in dpath for i in pi if i is not None ]
    mpath = [ [ G.edges[e]['relation'] if not ((e[0] in src_ids and e[1] in src_ids) or (e[0] in dst_ids and e[1] in dst_ids)) and e[1] > src_ids[0] and e[0] < dst_ids[0] else None, G.nodes[e[1]]['token'] if not (e[1] in src_ids or e[1] < src_ids[0] or e[1] in dst_ids or e[1] > dst_ids[-1]) else None ] for e in epath ]
    mpath = [ pi for pi in mpath if pi[0] is not None ]
    mpath = [ i for pi in mpath for i in pi if i is not None ]
  except (nx.NetworkXNoPath, nx.NodeNotFound):
    dpath = [ '-=*NO-PATH*=-' ]
    mpath = [ '-=*NO-PATH*=-' ]
    epath = [ ]
  return dpath, mpath, epath, G