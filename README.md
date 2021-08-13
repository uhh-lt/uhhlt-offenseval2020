#  UHH-LT at SemEval-2020 Task 12

## UHH-LT at SemEval-2020 Task 12: Fine-Tuning of Pre-Trained Transformer Networks for Offensive Language Detection

The implemenation presented here is part of the SemEval-2020 Task 12 on Multilingual OffensiveLanguage Identification in Social Media (OffensEval-2020). The Language Technology team (UHH-LT) from Hamburg University was ranked 1st on the English subtask A. The team fine-tuned different transformer models on the OLID training data and then combined into an ensemble. 

This repository contains the implementation of our approach proposed in the paper:

> Wiedemann, G., Yimam, S., Biemann, C. (2020): [UHH-LT at SemEval-2020 Task 12: Fine-Tuning of Pre-Trained Transformer Networks for Offensive Language Detection. Proceedings of the Fourteenth Workshop on Semantic Evaluation](https://www.aclweb.org/anthology/2020.semeval-1.213)


Fine-tuning of pre-trained transformer networks such as BERT yield state-of-the-art results for text classification tasks. Typically, fine-tuning is performed on task-specific training datasets in a supervised manner. One can also fine-tune in unsupervised manner beforehand by further pre-training the masked language modeling (MLM) task. Hereby, in-domain data for unsupervised MLM resembling the actual classification target dataset allows for domain adaptation of the model. In this paper, we compare current pre-trained transformer networks with and without MLM fine-tuning on their performance for offensive language detection. Our MLM fine-tuned RoBERTa-based classifier officially ranks 1st in the SemEval 2020 Shared Task 12 for the English language. Further experiments with the ALBERT model even surpass this result.

## Subtask A
The subtask A from OffensEval is Offensive Language Identification in English Language

## UHH-LT Methodology
We investigated two questions regarding the fine-tuning of pre-trained transformers: 
1. Which pre-trained models perform best on the 2020 OLD shared task?
2. How much language model fine-tuning on in-domain data prior to classification fine-tuning improves the performance of the best model?

For detailed explanations and findings, please refer our [paper](https://www.aclweb.org/anthology/2020.semeval-1.213.pdf). 

### Model Selection
We tested the following transformer-based pre-trained models for the OffensEval2020 OLD shared task.

1. BERT  –  Bidirectional  Encoder  Representations  from  Transformers
2. RoBERTa – A Robustly Optimized BERT Pretraining Approach
3. XLM-RoBERTa – XLM-R
4. ALBERT – A Lite BERT for Self-supervised Learning of Language Representations

This repository contains the implementation for the best ensemble which is based on the ALBERT model

The following sections explains in details as how to use them. 

## Requirements  (add links here and refine the list)
* Python3.5 or above
* [Flair](https://github.com/zalandoresearch/flair)
* TensorboardX
* Spacy


## Datasets
This section will guide you through the steps to use the datasets that were used in the paper to reproduce the results:

* The path to the datasets is "datasets/OffensEval20"

The util datasets.py has to imported and the required dataset can be loaded as follows:
```
import datasets as ds

For Task A
dataset = ds.OffensEvalData2020A(path = 'datasets/OffensEval20', n_max=-1)
s_train, s_test = dataset.getData()

For Task B
dataset = ds.OffensEvalData2020B(path = 'datasets/OffensEval20', n_max=-1)
s_train, s_test = dataset.getData()

```


## Using the Notebook

The notebook oe20_classification.ipynb contains the implementation for subtask A: Offensive Language Detection. 
(The steps are mapped in the notebook)

1. Ensure that all the requirements are satisfied by running the pip
2. Select the model by changing the index to the required one from the dictionary of model name:

                    1: 'albert-base-v1',
                    2: 'albert-large-v1',
                    3: 'albert-xlarge-v1',
                    4: 'albert-xxlarge-v1',    
                    5: 'albert-base-v2', 
                    6: 'albert-large-v2', 
                    7: 'albert-xlarge-v2', 
                    8: 'albert-xxlarge-v2'
3. Load the model and train them on the dataset loaded from the previous section. Refer our paper for training details. 
4. Save the pre-trained check points under the respective folders
5. Evaluate the pre-trained model on the test set
6. Load this pre-trained models and ensemble them
7. Using the majority vote option, the predictions are computed
8. The Precision, Recall and F1 scores for labels {NOT, OFF} and overall Macro F1 and Accuracy are determined. 

Note: The experiments were re-run and the results are slightly different from the orginal results in the paper. We observed that the environment in which the models are trained influences the evaluation and prediction results. 

## Citation


If you use these resources and models, please cite the following paper:

```
@InProceedings{Wiedemann2020,
  author    = {Gregor Wiedemann and Seid Muhie Yimam and Chris Biemann },
  title     = {UHH-LT at SemEval-2020 Task 12: Fine-Tuning of Pre-Trained Transformer Networks for Offensive Language Detection},
  booktitle = {Proceedings of the 14th International Workshop on Semantic Evaluation},
  year      = {2020},
  address   = {Barcelona, Spain (Online)},
  pages     = {1638–1644}
}
```
