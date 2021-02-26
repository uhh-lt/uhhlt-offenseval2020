from abc import ABC, abstractmethod
from typing import Union, List, Dict, Tuple

from flair.data import Sentence, Label
from flair.embeddings import Embeddings
from flair.models import SequenceTagger

from embedding import TransformerDocumentEmbeddings

import regex
import math
import torch
import numpy as np

import os
import sys
import pickle
import pandas as pd
from copy import copy

from collections import Counter, defaultdict
from sklearn.metrics import classification_report, cohen_kappa_score, roc_auc_score, roc_curve, precision_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

#from tqdm.notebook import tqdm
from tqdm.autonotebook import tqdm

from basetypeutils import DataSet, StructurePairResult, ResultSet, RetrievalResult, MaskableToken, MaskableSentence, MaskedStructure, LabeledData
import spacyutils

# regexes
punctuation_marks = regex.compile(r'^\p{P}+?$', regex.UNICODE)

# cuda or cpu?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MaskedDataset(object):

    def __init__(self, data: DataSet):
        self.use_tokenizer: bool = True
        self.ids, self.texts, self.labels = zip(*data)
        self.sentences: List[Sentence] = None
        self.structures: List[List[MaskedStructure]] = None
        self.is_contextualized: bool = False

    def __len__(self):
        assert len(self.ids) == len(self.texts), "Number of IDs unequal to number of texts"
        assert len(self.ids) == len(self.labels), "Number of IDs unequal to number of labels"
        return len(self.ids)

    def get_stratified_subsets(self, size: Union[int, float], result:str = 'subset') -> Union[Tuple[object, object], object]:
      '''
      Parameters
      ----------
      test_size : Union[int, float]
        The size of the subset either absolut or relative to the current dataset.
      result : str, optional
        What to return: one of {'subset', 'remainder', 'both'} the default is 'subset'.
      Returns
      -------
      Tuple[object, object]
        The resulting subset or remainder of the MaskedDataset or both: (subset, remainder).
      '''

      try:
        remainder_idxs, subset_idxs = train_test_split(range(len(self.ids)), test_size=size, random_state=42, stratify=self.labels)
      except:
        print('Stratified sampling failed, sampling randomly.')
        remainder_idxs, subset_idxs = train_test_split(range(len(self.ids)), test_size=size, random_state=42)

      stratfied_subset = None
      stratified_remainder = None

      if 'subset' == result or 'both' == result:
        stratfied_subset = copy(self)
        stratfied_subset.ids = []
        stratfied_subset.texts = []
        stratfied_subset.labels = []
        if self.sentences is not None:
          stratfied_subset.sentences = []
        if self.structures is not None:
          stratfied_subset.structures = []
        for i in subset_idxs:
          stratfied_subset.ids.append(self.ids[i])
          stratfied_subset.texts.append(self.texts[i])
          stratfied_subset.labels.append(self.labels[i])
          if self.sentences is not None:
            stratfied_subset.sentences.append(self.sentences[i])
          if self.structures is not None:
            stratfied_subset.structures.append(self.structures[i])

      if 'remainder' or 'both' == result:
        stratified_remainder = copy(self)
        stratified_remainder.ids = []
        stratified_remainder.texts = []
        stratified_remainder.labels = []
        if self.sentences is not None:
          stratified_remainder.sentences = []
        if self.structures is not None:
          stratified_remainder.structures = []
        for i in remainder_idxs:
          stratified_remainder.ids.append(self.ids[i])
          stratified_remainder.texts.append(self.texts[i])
          stratified_remainder.labels.append(self.labels[i])
          if self.sentences is not None:
            stratified_remainder.sentences.append(self.sentences[i])
          if self.structures is not None:
            stratified_remainder.structures.append(self.structures[i])

      return stratfied_subset, stratified_remainder

    def setLabels(self, labels):
        pass

    def contextualize(self, embedding_model: Tuple[Embeddings, int, int]):
        # run embedding
        self.sentences = [Sentence(s, use_tokenizer=self.use_tokenizer) for s in self.texts]
        batch_size = 32
        total_batches = math.ceil(len(self.sentences) / batch_size)
        for batch in tqdm(divide_into_batches(self.sentences, size = batch_size),
                          total = total_batches,
                          desc = 'Calculate token embeddings'):
            embedding_model[0].embed(batch)
        for sentence in self.sentences:
            for token in sentence:
                token.__class__ = MaskableToken
                token._init_MaskableToken()
                # sum n_layer final layers for token representation
                token.sum_embedding = token.embedding.view(embedding_model[1], embedding_model[2]).sum(axis=0).cpu()
                token.clear_embeddings()
        self.is_contextualized = True


    def cls_embed(self, embedding_model: TransformerDocumentEmbeddings):
        # run CLS embedding after contextualized embedding (yes, it is redundant work, but ...)
        batch_size = 32
        total_batches = math.ceil(len(self.sentences) / batch_size)
        for batch in tqdm(divide_into_batches(self.sentences, size = batch_size),
                          total = total_batches,
                          desc = 'Calculate CLS embeddings'):
            embedding_model.embed(batch)


    def aggregateMasks(self):
        self.structures = []
        for i, sentence in enumerate(self.sentences):
            # collect
            mask_embs: Dict[int, [torch.tensor]] = {}
            mask_texts: Dict[int, [str]] = {}
            for token in sentence:
                # ignore 0 mask
                if token.mask_id > 0:
                    if token.whitespace_after:
                        ws = " "
                    else:
                        ws = ""
                    if token.mask_id in mask_embs:
                        mask_embs[token.mask_id].append(token.sum_embedding)
                        mask_texts[token.mask_id] += token.text + ws
                    else:
                        mask_embs[token.mask_id] = [token.sum_embedding]
                        mask_texts[token.mask_id] = token.text + ws
            # aggregate
            s_masked_structures = []
            for mask_id in sorted(mask_embs.keys()):
                structure_embeddings = torch.mean(torch.stack(mask_embs[mask_id], dim=0), axis=0)
                structure: MaskedStructure = MaskedStructure(
                    text = mask_texts[mask_id],
                    context = self.texts[i],
                    embedding = structure_embeddings,
                    label = self.labels[i],
                    id = self.ids[i]
                )
                s_masked_structures.append(structure)
            self.structures.append(s_masked_structures)

    @staticmethod
    def getAnnotatedSentence(masked_sentence):
      # tokens and simple masks
      text_and_simple_masks = ' '.join(['/'.join([f'{token.text}', '', f'''{token.getMask() if token.getMask() > 0 else ''}''']) for token in masked_sentence])
      # collect extended mask annotations
      extended_mask_annotations = ''
      if isinstance(masked_sentence, MaskableSentence):
        extended_mask_annotations = ' '.join([f'/{tokenid+1}-{tokenid+1}//{k}/' for k, v in masked_sentence.properties.items() for tokenid in v])
      return (text_and_simple_masks + ' ' + extended_mask_annotations).strip()


#     def getMaskedStructures(self) -> List[MaskedStructure]:
#         assert self.is_contextualized, "Text data is not embedded. Call contextualized() first."
#         labeled_instances: List[MaskedStructure] = []
#         for sentence_structures in self.structures:
#             for structure in sentence_structures:
#                 assert structure.label, "No label present for structure: %s" % structure
#             labeled_instances.extend(sentence_structures)
#         return labeled_instances


    def getNearestNeighbors(self, candidates: LabeledData, target_label: str = None, k: int = sys.maxsize) -> List[ResultSet]:
        assert self.structures, "Embeddings have not been aggregated along masks."

        similarities = []

        # sort labels increasing order
        label_list = np.array(list(candidates.keys()))
        counts = [len(candidates[label]) for label in label_list]
        label_list = label_list[np.argsort(counts)]
        # set target_label as first element in label_list
        if target_label is not None:
            assert target_label in label_list, "Target label %s not found in candidate data"
            new_label_list = [target_label] + [l for l in label_list if l != target_label]
            label_list = new_label_list

        neighbors_per_label = {}
        for label in label_list:
            similarities = []
            neighbor_tensor = torch.stack([structure.embedding for structure in candidates[label]], dim=0).cuda()
            # normalize
            neighbors_per_label[label] = neighbor_tensor / neighbor_tensor.norm(dim=1)[:, None]

        for i, structure in enumerate(tqdm(self.structures, desc='Calulating similarities')):
            sim_per_label = []
            similarity_stack = []
            candidates_stack = []
            for label in label_list:
                if not label in neighbors_per_label:
                  continue
                neighbor_tensor_norm = neighbors_per_label[label]
                struct_embs_list = [s.embedding for s in structure]
                structure_embs = torch.stack(struct_embs_list, dim=0).cuda()
                structure_embs_norm = structure_embs / structure_embs.norm(dim=1)[:, None]
                similarity = torch.mm(neighbor_tensor_norm, structure_embs_norm.transpose(0,1))
                similarity_stack.append(similarity)
                candidates_stack.extend(candidates[label])

                top_k_sim, _ = similarity.sort(dim=0, descending=True)
                # average vectors of most similar structures per label
                sim_per_label.append(top_k_sim[:min(top_k_sim.size(0), k),:].mean(0))

            if len(label_list) == 2:
                # subtract from first label
                difference = sim_per_label[0] - torch.stack(sim_per_label[1:], dim=0).sum(0)
                impact, sorted_structure = difference.sort(descending = True)

                # print("--")
                # for j, s_debug in enumerate(sorted_structure):
                #     print("%.3f" % float(impact[j]) + "\t" + self.structures[i][s_debug].text)

                selected_structure = sorted_structure[0]
            else:
                # multi class case
                highest_impact = float("-inf")
                for j in range(len(label_list)):
                    other_similarities = [sim_per_label[m] for m in range(len(label_list)) if m != j]
                    difference = sim_per_label[j] - torch.stack(other_similarities, dim=0).sum(0)
                    impact, sorted_structure = difference.sort(descending = True)
                    if impact[0] > highest_impact:
                        highest_impact = impact[0]
                        selected_structure = sorted_structure[0]
#                         # debug output
#                         highest_impact_list = impact
#                         highest_sorted_structure = sorted_structure
#                         selected_label = label_list[j]
#                 print("--")
#                 print(selected_label)
#                 for m, s_debug in enumerate(highest_sorted_structure):
#                     print("%.3f" % float(highest_impact_list[m]) + "\t" + self.structures[i][s_debug].text)

            sim_to_selected = torch.cat(similarity_stack, dim=0)[:, selected_structure]

            # sim, test.label, train.label, test.text, train.text, test.context, train.context, test.id, train.id
            comparison_structures = sim_to_selected.argsort(descending=True)[:min(sim_to_selected.size(0), k * 3)] # there might be more than 1 nn structure per instance, but we want to filter per instance later so we allow here up to 3 times more (retrieving all and filter k ids later slows down the process too much, but k * 3 is safe)
            comparison_structures = [(int(r), selected_structure) for r in comparison_structures]

            collected_ids = set()
            res_tuples: ResultSet = []
            for pair in comparison_structures:
                trainsample_id = candidates_stack[pair[0]].id
                if trainsample_id in collected_ids:
                    # collect only nearest structure per trainsample
                    continue
                res_tuples.append((
                    float(sim_to_selected[pair[0]]), # sim(testsample, candidate)
                    self.structures[i][pair[1]].label, # testsample true label
                    candidates_stack[pair[0]].label, # NN candidate label
                    self.structures[i][pair[1]].text.strip(), # testsample masked text
                    candidates_stack[pair[0]].text.strip(),  # candidate masked text
                    self.structures[i][pair[1]].context,  # testsample sentence
                    candidates_stack[pair[0]].context,  # candidate sentence
                    self.structures[i][pair[1]].id, # testsample id
                    trainsample_id # candidate id
                ))
                collected_ids.add(trainsample_id)
                if len(res_tuples) == k:
                    break
            if len(res_tuples) < k:
                print("WARNING: less than %d instances retrieved instead of %d" % (len(res_tuples), k))

            similarities.append(res_tuples)

            del similarity
            del similarity_stack
            del sim_per_label
            del structure_embs
            del structure_embs_norm

        return similarities


    def reorderStructures(self, candidates: LabeledData, target_label: str = None, k: int = 10):
        assert self.structures, "Embeddings have not been aggregated along masks."

        self.impacts = []

        # sort labels increasing order
        label_list = np.array(list(candidates.keys()))
        counts = [len(candidates[label]) for label in label_list]
        label_list = label_list[np.argsort(counts)]
        # set target_label to front
        if target_label is not None:
            assert target_label in label_list, "Target label %s not found in candidate data"
            new_label_list = [target_label] + [l for l in label_list if l != target_label]
            label_list = new_label_list

        neighbors_per_label = {}
        for label in label_list:
            similarities = []
            neighbor_tensor = torch.stack([structure.embedding for structure in candidates[label]], dim=0).cuda()
            neighbor_tensor_norm = neighbor_tensor / neighbor_tensor.norm(dim=1)[:, None]
            neighbors_per_label[label] = neighbor_tensor_norm

        for i, structure in enumerate(tqdm(self.structures, desc='Calulating similarities')):
            sim_per_label = []
            similarity_stack = []
            candidates_stack = []
            for label in label_list:
                neighbor_tensor_norm = neighbors_per_label[label]
                struct_embs_list = [s.embedding for s in structure]
                structure_embs = torch.stack(struct_embs_list, dim=0).cuda()
                structure_embs_norm = structure_embs / structure_embs.norm(dim=1)[:, None]
                similarity = torch.mm(neighbor_tensor_norm, structure_embs_norm.transpose(0,1))
                similarity_stack.append(similarity)
                candidates_stack.extend(candidates[label])

                top_k_sim, _ = similarity.sort(dim=0, descending=True)
                # average vectors of most similar structures per label
                sim_per_label.append(top_k_sim[:k,:].mean(0))

            differences = []
            if len(label_list) == 2:
                # subtract from first label
                difference = sim_per_label[0] - torch.stack(sim_per_label[1:], dim=0).sum(0)
                differences.append(difference)
            else:
                # multi class case
                for j in range(len(label_list)):
                    other_similarities = [sim_per_label[m] for m in range(len(label_list)) if m != j]
                    difference = sim_per_label[j] - torch.stack(other_similarities, dim=0).sum(0)
                    differences.append(difference)

            max_values, max_ind = torch.stack(differences, dim=0).max(dim=0)
            impact, sorted_structure = max_values.sort(descending = True)

            self.structures[i] = [self.structures[i][j] for j in sorted_structure]
            self.impacts.append(impact)


    def getLabeledData(self) -> LabeledData:
        assert self.is_contextualized, "Text data is not embedded. Call contextualize() first."
        # collect labels
        labeled_instances: LabeledData = {}
        for sentence_structures in self.structures:
            for structure in sentence_structures:
                if structure.label not in labeled_instances:
                    labeled_instances[structure.label] = []
                # skip over 0 embeddings
                if structure.embedding.nonzero().size(0) > 0: 
                    labeled_instances[structure.label].append(structure)
        # remove labels that had only zero embeddings
        for label in list(labeled_instances.keys()):
            if not len(labeled_instances[label]):
              del labeled_instances[label]
        return labeled_instances


def filterNearestNeighborsByUniqueID(results : List[ResultSet]):
    fresults = []
    for result in results: # for each query result
        collected_ids = set()
        fresults.append([])
        for neighbor_result in result:
            _, _, _, _, _, _, _, _, trainsample_id = neighbor_result
            if trainsample_id in collected_ids:
                continue
            collected_ids.add(trainsample_id)
            fresults[-1].append(neighbor_result)
    return fresults


class MaskedSpacyDataset(MaskedDataset):

    def __init__(self, spacy_dataset, embedding_model: Tuple[Embeddings, int, int], save_contextualized_sentences:str = None, *args, aggregation='dependency-path', **kwargs):
        super().__init__(spacy_dataset.dataset)
        self.embedding_model = embedding_model
        self.spacy_dataset = spacy_dataset
        self.save_contextualized_sentences = None
        self.aggregation = aggregation
        if save_contextualized_sentences is not None:
          if isinstance(save_contextualized_sentences, str):
              self.save_contextualized_sentences = save_contextualized_sentences
          else:
              if save_contextualized_sentences:
                self.save_contextualized_sentences = f'''{os.path.basename(str(self.spacy_dataset.save_to_file)).replace('.pkl', '')}__{self.embedding_model[0].__class__.__name__}__token_embeddings.pkl'''

    # override
    def contextualize(self, embedding_model: Tuple[Embeddings, int, int] = None):
        if self.embedding_model is None:
            self.embedding_model = embedding_model
        # check if saved embeddings are available
        if self.save_contextualized_sentences is not None and os.path.isfile(self.save_contextualized_sentences):
            print(f'''Loading embeddings for sentences from '{self.save_contextualized_sentences}' ''')
            with open(self.save_contextualized_sentences, 'rb') as f:
                self.sentences = pickle.load(f)
            if len(self.sentences) != len(self.ids):
                print('Inconsistent sentences. Reloading and not saving...')
                self.save_contextualized_sentences = None
                self.is_contextualized = False
                return self.contextualize()
        else:
          # get embeddings
          self.sentences = [ MaskableSentence(text=sent, use_tokenizer=spacyutils.build_pandas_id_tokenizer(strid, self.spacy_dataset.get_spacy_doc_for_id) ) for sent, strid in zip(self.texts, self.ids) ]
          batch_size = 8
          total_batches = math.ceil(len(self.sentences) / batch_size)
          for batch in tqdm(divide_into_batches(self.sentences, size = batch_size),
                            total = total_batches,
                            desc = 'Calculate embeddings'):
              self.embedding_model[0].embed(batch)

          if self.save_contextualized_sentences is not None:
              print(f'''saving embeddings from sentences in '{self.save_contextualized_sentences}' ''')
              with open(self.save_contextualized_sentences, 'wb') as f:
                  pickle.dump(self.sentences, f)

        for sentence in self.sentences:
            for token in sentence:
                token.__class__ = MaskableToken
                token._init_MaskableToken()
                contains_nan = torch.isnan(token.embedding).sum()
                is_zero_vec = token.embedding.sum() == 0
                if contains_nan or is_zero_vec:
                  for vec in token.get_each_embedding():
                    vec.normal_(mean=0.0, std=1e-12) # make zeros with some tiny std-dev
                # sum n_layer final layers for token representation
                token.sum_embedding = token.embedding.view(self.embedding_model[1], self.embedding_model[2]).sum(axis=0)
        self.is_contextualized = True


    def aggregateMasks(self):
      return {
        'dependency-path': self.aggregate_dependencypath_masks,
        'tag': self.aggregateMasks_by_token_tag,
        'id': super().aggregateMasks
      }[self.aggregation]()

    def aggregate_dependencypath_masks(self):
        self.structures = []

        for i, sentence in enumerate(self.sentences):
            # collect dependency path structures

            # prepare structured tensor
            emdim = self.embedding_model[2] * self.embedding_model[1]
            x = torch.zeros(4 * emdim).normal_(mean=0.0, std=1e-12)

            # set averaged e1 tokens
            z = torch.zeros(emdim).normal_(mean=0.0, std=1e-12)
            if len(sentence.properties['e1']):
              for j in sentence.properties['e1']:
                  z += sentence[j].embedding
              x[:emdim] = z / len(sentence.properties['e1'])

            # set averaged e2 tokens
            z = torch.zeros(emdim).normal_(mean=0.0, std=1e-12)
            if len(sentence.properties['e2']):
              for j in sentence.properties['e2']:
                  z += sentence[j].embedding
              x[emdim:2*emdim] = z / len(sentence.properties['e2'])

            # set averaged path tokens
            z = torch.zeros(emdim).normal_(mean=0.0, std=1e-12)
            if len(sentence.properties['part-of-path-inbetween']):
              for j in sentence.properties['part-of-path-inbetween']:
                  z += sentence[j].embedding
              x[2*emdim:3*emdim] = z / len(sentence.properties['part-of-path-inbetween'])

            # set averaged path tokens which are not between e1, and e2
            on_path_but_not_inbetween = set(sentence.properties['part-of-path']) - set(sentence.properties['part-of-path-inbetween'])
            z = torch.zeros(emdim).normal_(mean=0.0, std=1e-12)
            if len(on_path_but_not_inbetween):
              for j in on_path_but_not_inbetween:
                  z += sentence[j].embedding
              x[3*emdim:4*emdim] = z / len(on_path_but_not_inbetween)

            # set sentence embedding
            # em_cls = sentence.embedding
            # print(em_cls.shape)
            # x[4*emdim:] = em_cls

            # create structure
            structure: MaskedStructure = MaskedStructure(
              text = ' '.join(map(lambda j: sentence[j].text, sentence.properties['e1'] + sentence.properties['part-of-path-inbetween'] + sentence.properties['e2'])),
              context = self.texts[i],
              embedding = x,
              label = self.labels[i],
              id = self.ids[i]
            )
            self.structures.append([ structure ])

    def aggregateMasks_by_token_tag(self):
        self.structures = []
        for i, sentence in enumerate(self.sentences):
            # collect
            mask_embs: Dict[str, [torch.tensor]] = {}
            mask_texts: Dict[str, [str]] = {}
            for token in sentence:
                maskval = token.get_tag('maskid').value
                # ignore empty mask values and ignore 0 mask
                if len(maskval) > 0 and maskval != '0':
                    if maskval in mask_embs:
                        mask_embs[maskval].append(token.sum_embedding)
                        mask_texts[maskval] += ' ' + token.text
                    else:
                        mask_embs[maskval] = [token.sum_embedding]
                        mask_texts[maskval] = token.text
            # aggregate
            s_masked_structures = []
            for mask_id in sorted(mask_embs.keys()):
                structure_embeddings = torch.mean(torch.stack(mask_embs[mask_id], dim=0), axis=0)
                #print(structure_embeddings.shape)
                structure: MaskedStructure = MaskedStructure(
                    text = mask_texts[mask_id],
                    context = self.texts[i],
                    embedding = structure_embeddings,
                    label = self.labels[i],
                    id = self.ids[i]
                )
                s_masked_structures.append(structure)
            self.structures.append(s_masked_structures)


class AbstractMaskingStrategy(ABC):
    def __init__(self, embedding_model: Tuple[Embeddings, int, int], cls_embedding_model: TransformerDocumentEmbeddings):
        self.embedding_model = embedding_model
        self.cls_embedding_model = cls_embedding_model

    def generateMask(self, dataset: MaskedDataset, cls_mask: bool = False):
        if not dataset.is_contextualized:
            print("Computing contextualized embeddings for %d instances" % len(dataset.texts))
            dataset.contextualize(self.embedding_model)

        if self.cls_embedding_model is not None and cls_mask:
            print("Computing CLS embeddings for %d instances" % len(dataset.texts))
            dataset.cls_embed(self.cls_embedding_model)

        print("Aggregating embeddings along mask ids greater 0")
        # init all mask ids to 0
        for s in dataset.sentences:
            for t in s:
                if isinstance(t, MaskableToken):
                    t.setMask(0)
                t.add_tag('maskid', '0')
        # set masks according to some strategy
        self.setMasks(dataset)
        dataset.aggregateMasks()

    @abstractmethod
    def setMasks(self, dataset: MaskedDataset):
        pass


class BaselineCLS(AbstractMaskingStrategy):
    def setMasks(self, dataset: MaskedDataset):
        for sentence in dataset.sentences:
            for token in sentence:
                token.setMask(0)
            sentence.add_token("[CLS]")
            cls_token = sentence[-1]
            cls_token.__class__ = MaskableToken
            cls_token._init_MaskableToken()
            cls_token.sum_embedding = sentence.embedding
            cls_token.setMask(1)


class BaselineBoW(AbstractMaskingStrategy):
    def setMasks(self, dataset: MaskedDataset):
        for sentence in dataset.sentences:
            for token in sentence:
                token.setMask(1)

class SingleToken(AbstractMaskingStrategy):
    def setMasks(self, dataset: MaskedDataset):
        for sentence in dataset.sentences:
            i = 0
            for token in sentence:
                i += 1
                token.setMask(i)


class SingleWord(AbstractMaskingStrategy):
    def setMasks(self, dataset: MaskedDataset):
        for sentence in dataset.sentences:
            i = 0
            for token in sentence:
                if not regex.match(punctuation_marks, token.text.strip()):
                    i += 1
                    token.setMask(i)
            # fallback option: set first token to 1
            if i == 0:
                sentence[0].setMask(1)

class NonStopWord(AbstractMaskingStrategy):
    def __init__(self, embedding_model: Tuple[Embeddings, int, int], cls_embedding_model: TransformerDocumentEmbeddings):
        super().__init__(embedding_model, cls_embedding_model)
        with open("stopwords.txt", "r") as f:
            content = f.readlines()
        self.stopwords = [x.strip() for x in content]

    def setMasks(self, dataset: MaskedDataset):
        for sentence in dataset.sentences:
            i = 0
            for token in sentence:
                w = token.text.strip()
                if not regex.match(punctuation_marks, w) and w.lower() not in self.stopwords:
                    i += 1
                    token.setMask(i)
            # fallback option: set first token to 1
            if i == 0:
                sentence[0].setMask(1)


class Chunk(AbstractMaskingStrategy):
    def __init__(self, embedding_model: Tuple[Embeddings, int, int], cls_embedding_model: TransformerDocumentEmbeddings):
        super().__init__(embedding_model, cls_embedding_model)
        self.chunker = SequenceTagger.load('chunk')
        self.type_filter = 'PP'
        with open("stopwords.txt", "r") as f:
            content = f.readlines()
        self.stopwords = [x.strip() for x in content]
    def setMasks(self, dataset: MaskedDataset):
        chunked_sentences = self.chunker.predict(dataset.sentences)
        for sentence in dataset.sentences:
            i = 0
            for token in sentence:
                w = token.text.strip()
                chunk_tag = token.tags['np'].value
                chunk_type = token.tags['np'].value[2:]
                # skip pp chunks and puctuation marks
                if chunk_type == self.type_filter or regex.match(punctuation_marks, w) or (chunk_tag.startswith('S') and w.lower() in self.stopwords):
                    continue
                # beginning or single token
                if chunk_tag.startswith('B') or chunk_tag.startswith('S'):
                    # increase mask id
                    i += 1
                token.setMask(i)
                # print(token.mask_id, token.text)

            # fallback option: set first token to 1
            if i == 0:
                sentence[0].setMask(1)

class ChunkSW(AbstractMaskingStrategy):
    def __init__(self, embedding_model: Tuple[Embeddings, int, int], cls_embedding_model: TransformerDocumentEmbeddings):
        super().__init__(embedding_model, cls_embedding_model)
        self.chunker = SequenceTagger.load('chunk')
        self.type_filter = 'PP'
        with open("stopwords.txt", "r") as f:
            content = f.readlines()
        self.stopwords = [x.strip() for x in content]
    def setMasks(self, dataset: MaskedDataset):
        chunked_sentences = self.chunker.predict(dataset.sentences)
        for sentence in dataset.sentences:
            i = 0
            for token in sentence:
                w = token.text.strip()
                chunk_tag = token.tags['np'].value
                chunk_type = token.tags['np'].value[2:]
                # skip pp chunks and puctuation marks
                if chunk_type == self.type_filter or regex.match(punctuation_marks, w):
                    continue
                # beginning or single token
                if chunk_tag.startswith('B') or chunk_tag.startswith('S'):
                    # increase mask id
                    i += 1
                if w.lower() not in self.stopwords:
                    token.setMask(i)
                # print(token.mask_id, token.text)

            # fallback option: set first token to 1
            nothing_masked = True
            for t in sentence:
                if t.mask_id:
                    nothing_masked = False
            if nothing_masked:
                sentence[0].setMask(1)


class Chunk_byTag(AbstractMaskingStrategy):
    def __init__(self, data: DataSet):
        super().__init__(data)
        self.chunker = SequenceTagger.load('chunk')
        self.type_filter = 'PP'
        with open("stopwords.txt", "r") as f:
            content = f.readlines()
        self.stopwords = [x.strip() for x in content]
    def setMasks(self, dataset: MaskedDataset):
        chunked_sentences = self.chunker.predict(dataset.sentences)
        for sentence in dataset.sentences:
            i = 0
            for token in sentence:
                w = token.text.strip()
                chunk_tag = token.get_tag('np').value
                chunk_type = token.get_tag('np').value[2:]
                # skip pp chunks and puctuation marks
                if chunk_type == self.type_filter or regex.match(punctuation_marks, w) or (chunk_tag.startswith('S') and w.lower() in self.stopwords):
                    continue
                # beginning or single token
                if chunk_tag.startswith('B') or chunk_tag.startswith('S'):
                    # increase mask id
                    i += 1
                #token.setMask(i)
                token.add_tag('maskid', str(i))
                # print(token.mask_id, token.text)

            # fallback option: set first token to 1
            if i == 0:
                #sentence[0].setMask(1)
                sentence[0].add_tag('maskid', '1')


class ChunkSW_byTag(AbstractMaskingStrategy):
    def __init__(self, data: DataSet):
        super().__init__(data)
        self.chunker = SequenceTagger.load('chunk')
        self.type_filter = 'PP'
        with open("stopwords.txt", "r") as f:
            content = f.readlines()
        self.stopwords = [x.strip() for x in content]
    def setMasks(self, dataset: MaskedDataset):
        chunked_sentences = self.chunker.predict(dataset.sentences)
        for sentence in dataset.sentences:
            i = 0
            for token in sentence:
                w = token.text.strip()
                chunk_tag = token.get_tag('np').value
                chunk_type = token.get_tag('np').value[2:]
                # skip pp chunks and puctuation marks
                if chunk_type == self.type_filter or regex.match(punctuation_marks, w):
                    continue
                # beginning or single token
                if chunk_tag.startswith('B') or chunk_tag.startswith('S'):
                    # increase mask id
                    i += 1
                if w.lower() not in self.stopwords:
                    # token.setMask(i)
                    token.add_tag('maskid', str(i))
                # print(token.mask_id, token.text)

            # fallback option: set first token to 1
            nothing_masked = True
            for t in sentence:
                if t.mask_id:
                    nothing_masked = False
            if nothing_masked:
                # sentence[0].setMask(1)
                sentence[0].add_tag('maskid', '1')


class DependencyPath(AbstractMaskingStrategy):

    def __init__(self):
        super().__init__(None, None)

    #@override
    def generateMask(self, dataset: MaskedSpacyDataset):
        super().generateMask(dataset)

    #@override
    def setMasks(self, dataset: MaskedSpacyDataset):
        for i, sentence in enumerate(dataset.sentences):

            sentence.properties['part-of-path-inbetween'] = []
            sentence.properties['part-of-path'] = []
            sentence.properties['e1'] = []
            sentence.properties['e2'] = []

            # get the parsed sentence
            extrnl_id, extrnl = dataset.spacy_dataset.getrow('strid', dataset.ids[i])

            e1_indices = range(*extrnl.offset_spacy_e1)
            for j in e1_indices:
                sentence[j].add_tag('emask', 'e1')
                sentence.properties['e1'].append(j)

            e2_indices = range(*extrnl.offset_spacy_e2)
            for j in e2_indices:
                sentence[j].add_tag('emask', 'e2')
                sentence.properties['e2'].append(j)

            psentence = extrnl.spacydoc
            sent_start = extrnl.offset_spacy_sentence
            dpath, mpath, _, _ = spacyutils.get_shortest_dependency_path(
              psentence,
              list(map(lambda k: k + sent_start, e1_indices)),
              list(map(lambda k: k + sent_start, e2_indices))
            )

            for node in dpath:
                if not isinstance(node, spacyutils.SpacyToken):
                    continue # dependency relation string
                sentence[node.i - sent_start].add_tag('amask', 'part-of-path')
                sentence.properties['part-of-path'].append(node.i - sent_start)

            for node in mpath:
                if not isinstance(node, spacyutils.SpacyToken):
                    continue # dependency relation string
                sentence[node.i - sent_start].add_tag('pmask', 'part-of-path-inbetween')
                sentence.properties['part-of-path-inbetween'].append(node.i - sent_start)

# functions
# ----------------------------------------------


# decision by argmax(count) of labels from k NN
def aggregateSentenceClassification(result: StructurePairResult) -> RetrievalResult:
    _, true_labels, pred_labels, test_structure, _, test_context, train_context, _, _ = zip(*result)
    most_frequent_label = Counter(pred_labels).most_common(1)[0]
    confidence = most_frequent_label[1] / len(result)
    sentence_classification = (
        confidence,
        most_frequent_label[0],
        true_labels[0],
        test_structure[0],
        test_context[0],
        train_context[0])
    return sentence_classification

# decision by argmax(sum) of labels from k NN
def aggregateSentenceClassificationWeighted(result: StructurePairResult) -> RetrievalResult:
    sim, true_labels, pred_labels, test_structure, _, test_context, train_context, _, _ = zip(*result)
    label_count = defaultdict(int)
    label_sum = defaultdict(float)
    total_sim = 0
    for i in range(len(pred_labels)):
        label_count[pred_labels[i]] += 1
        label_sum[pred_labels[i]] += sim[i]
        total_sim += sim[i]
    highest_confidence = 0
    best_label = None
    for k in label_sum.keys():
        label_sum[k] = label_sum[k] / total_sim
        if label_sum[k] > highest_confidence:
            highest_confidence = label_sum[k]
            best_label = k
    sentence_classification = (
        highest_confidence,
        best_label,
        true_labels[0],
        test_structure[0],
        test_context[0],
        train_context[0])
    return sentence_classification


def precision_at_k(result : ResultSet, pos_label : str) -> float:
    # retrieval evaluation is binary per definition
    sim, query_labels, retrieved_labels, test_structure, _, test_context, train_context, _, _ = zip(*result)
    # binary_labels = [l if l == pos_label else "OTHER" for l in retrieved_labels]
    # return precision_score(binary_labels, query_labels, pos_label=pos_label)
    precision = sum([1 if query_labels[i] == retrieved_labels[i] else 0 for i in range(len(query_labels))])
    if precision:
        precision = precision / len(query_labels)
    return precision

def average_precision(result : ResultSet, pos_label: str) -> float:
    # retrieval evaluation is binary per definition
    sim, query_labels, retrieved_labels, test_structure, _, test_context, train_context, _, _ = zip(*result)
    # binary_labels = [l if l == pos_label else "OTHER" for l in retrieved_labels]
    all_precisions = []
    for i in range(len(query_labels)):
        if retrieved_labels[i] == pos_label:
            precision = sum([1 if query_labels[j] == retrieved_labels[j] else 0 for j in range(i + 1)]) ### FIXME: why range(i+1)? isn't that effectively precision@k+1? this could be in index out-of-bounds if query_labels[k+1] == retrieved_labels[k+1]?!
            precision = precision / (i + 1)
            all_precisions.append(precision)
    if all_precisions:
        res = np.mean(all_precisions)
    else:
        res = 0
    return res

def evaluation_scores(results : List[ResultSet]):
    # retrieval results
    label_set = set([r[0][1] for r in results])
    retrieval_results = {'kpr' : {}, 'map' : {}}
    for pos_label in label_set:
        print("Retrieval evaluation for label: %s" % pos_label)
        mean_k_precision = np.mean([precision_at_k(r, pos_label) for r in results if r[0][1] == pos_label])
        mean_average_precision = np.mean([average_precision(r, pos_label) for r in tqdm(results, desc="Calculating average precision") if r[0][1] == pos_label])
        print("Precision at k = %d: %.3f" % (len(results[0]), mean_k_precision))
        print("Mean average precicsion: map = %.3f\n" % mean_average_precision)
        retrieval_results['kpr'][pos_label] = mean_k_precision
        retrieval_results['map'][pos_label] = mean_average_precision

    # aggregate pairs to sentence classifications
    classification_results = [aggregateSentenceClassification(r) for r in results]
    combined_metrics = classification_evaluation(classification_results)

    combined_metrics['kpr'] = retrieval_results['kpr']
    combined_metrics['map'] = retrieval_results['map']

    return combined_metrics, classification_results


def knn_prediction_and_retieval_result_tables(results : List[ResultSet]):
    # retrieval results
    k_max = -1
    neighbor_label_at_k_list = [] # collect neighbor label at each k
    knn_predictions_at_k_list = [] # collect knn prediction at each k
    prec_at_k_list = [] # collect precision at k values for each sample
    aprec_at_k_list = [] # collect precision at k values for each sample but only if the correct label was predicted
    for result in results: # for each query result
        _testsample_id_ = None
        _testsample_truelabel_ = None
        knn_predicted_label_at_k = []
        tp_at_k = []
        correct_at_k = []
        prec_at_k = []
        neighbor_label_at_k = []
        for (i, (sim, truelabel, neighborlabel, _, _, _, _, testsample_id, _)) in enumerate(result):
            if _testsample_id_ == None:
                _testsample_id_ = testsample_id
                _testsample_truelabel_ = truelabel
            assert _testsample_id_ == testsample_id, f"something's wrong. Expected test sample {_testsample_id_} but got {testsample_id}."
            assert _testsample_truelabel_ == truelabel, f"something's wrong. Expected true label {_testsample_truelabel_} for sample {_testsample_id_} but got {truelabel}."
            if i > 0:
              assert sim <= result[i-1][0], f"something's wrong. Expected orderderd list of neighbors by decreasing similarity."
            neighbor_label_at_k.append(neighborlabel)
            label_votes = Counter(neighbor_label_at_k)
            # collect the number of true positives at current k (for further reference)
            tp = label_votes[_testsample_truelabel_]
            tp_at_k.append(tp)
            prec_at_k.append(tp / len(tp_at_k))
            correct_at_k.append(1 if neighborlabel == truelabel else 0)
            # knn classification by max votes
            label_votes_ordered = label_votes.most_common()
            majority_labels = list(map(lambda x: x[0], filter(lambda x: label_votes_ordered[0][1] == x[1], label_votes_ordered))) # select only the most common predicted labels
            knn_prediction = majority_labels[np.random.randint(0, len(majority_labels))] # and choose a random label of the most common in case of ties
            knn_predicted_label_at_k.append(knn_prediction)
        if k_max < 0: # update k_max at the first test sample, then assert that every sample has the same k_max!
            k_max = len(knn_predicted_label_at_k)
        assert len(knn_predicted_label_at_k) == k_max, f"Something's wrong. Expected max {k_max} nearest neighbors, but got {len(knn_predicted_label_at_k)} for test sample id {_testsample_id_}."
        neighbor_label_at_k_list.append([ _testsample_id_, truelabel ] + neighbor_label_at_k)
        knn_predictions_at_k_list.append([ _testsample_id_, truelabel ] + knn_predicted_label_at_k)
        prec_at_k_list.append([ _testsample_id_, truelabel ] + prec_at_k)
        correct_at_k = np.array(correct_at_k)
        aprec_at_k = correct_at_k * np.array(prec_at_k)
        aprec_at_k_list.append([ _testsample_id_, truelabel ] + aprec_at_k.tolist())

    # prepare some dataframes for debugging purposes
    neighbor_labels_df = pd.DataFrame(neighbor_label_at_k_list, columns=['id','trueclass'] + list(map(lambda i: f'neighbor@k={i+1}', range(k_max))))
    knn_predictions_df = pd.DataFrame(knn_predictions_at_k_list, columns=['id','trueclass'] + list(map(lambda i: f'knn-pred@k={i+1}', range(k_max))))
    prec_at_k_df = pd.DataFrame(prec_at_k_list, columns=['id','trueclass'] + list(map(lambda i: f'prec@k={i+1}', range(k_max))))
    aprec_at_k_df = pd.DataFrame(aprec_at_k_list, columns=['id','trueclass'] + list(map(lambda i: f'aprec@k={i+1}', range(k_max))))

    # compute precision@k for each class
    per_class_prec_at_k = prec_at_k_df.groupby(['trueclass']).mean()
    # compute average precision and mean average precision (this is irrelevant and technically wrong)
    avg_ap = aprec_at_k_df.apply(lambda r: r[2:].mean(), axis=1).mean()
    # this should be the correct mAP score, but it is still missing all relevant documents
    mean_avg_prec = aprec_at_k_df.apply(lambda r: r[2:][r[2:]!=0].mean(), axis=1).mean() #np.mean(avgprec_list)
    # prepare to compile all values in one dataframe with respective score descriptions for debugging purposes
    map_values = [
      ['ALL', mean_avg_prec, avg_ap, prec_at_k_df.shape[0], np.nan ] + prec_at_k_df.iloc[:,2:].mean().tolist()
    ]
    per_class_map_columns = list(neighbor_labels_df.trueclass.unique())
    for classlabel in per_class_map_columns:
        per_class_aprec = aprec_at_k_df.loc[aprec_at_k_df.trueclass == classlabel]
        mean_avg_prec = per_class_aprec.apply(lambda r: r[2:][r[2:]!=0].mean(), axis=1).mean()
        avg_ap = per_class_aprec.apply(lambda r: r[2:].mean(), axis=1).mean()
        class_support_relevant = per_class_aprec.apply(lambda r: (r[2:]!=0).sum(), axis=1)
        class_support_relevant_val = class_support_relevant.iloc[0]
        if not (class_support_relevant == class_support_relevant_val).all(): # if not all of the rows have the same relevant support N k was chosen to be smaller than the entire dataset, and the map value is only approximated
            class_support_relevant_val = ';'.join(class_support_relevant.apply(str))
        map_values.append([classlabel, mean_avg_prec, avg_ap, per_class_aprec.shape[0], class_support_relevant_val] + per_class_prec_at_k.loc[classlabel].tolist())
    map_df = pd.DataFrame(map_values, columns=['class', 'mAP', 'aAP', 'support-test', 'support-relevant'] + prec_at_k_df.columns[2:].tolist())
    map_df.loc[0, 'support-relevant'] = map_df.loc[1:,'support-relevant'].sum()

    return neighbor_labels_df, knn_predictions_df, map_df


def classification_evaluation(results: List[RetrievalResult]):
    _, y_pred, y_true, _, _, _ = zip(*results)
    report = classification_report(y_true, y_pred, output_dict = True)
    kappa_value = cohen_kappa_score(y_true, y_pred)
    print(classification_report(y_true, y_pred, output_dict = False))
    print("\n\nCohens Kappa: ", kappa_value)
    report['kappa'] = kappa_value
    return(report)

    # # calculate scores
    # labels = set(y_pred + y_true)
    # label_dict = {}
    # for label in labels:
    #     label_dict[label] = len(label_dict)
    # y_pred_idx = [label_dict[label] for label in y_pred]
    # y_true_idx = [label_dict[label] for label in y_true]

    # lr_auc = roc_auc_score(y_true_idx, y_pred_idx)
    # # summarize scores
    # print('ROC AUC=%.3f' % (lr_auc))
    # # calculate roc curves
    # lr_fpr, lr_tpr, _ = roc_curve(y_true_idx, y_pred_idx)
    # # plot the roc curve for the model
    # pyplot.plot(lr_fpr, lr_tpr, marker='.', label='kNN')
    # # axis labels
    # pyplot.xlabel('False Positive Rate')
    # pyplot.ylabel('True Positive Rate')
    # # show the legend
    # pyplot.legend()
    # # show the plot
    # pyplot.show()

def divide_into_batches(l, size = 32):
    for i in range(0, len(l), size):
        yield l[i:i + size]