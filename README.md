# uhhlt-offenseval2020
## UHH-LT at SemEval-2020 Task 12: Fine-Tuning of Pre-Trained Transformer Networks for Offensive Language Detection

The implemenation presented here is part of the SemEval-2020 Task 12 on Multilingual OffensiveLanguage Identification in Social Media (OffensEval-2020). The Language Technology team (UHH-LT) from Hamburg University was ranked 1st on the English subtask A. The team fine-tuned different transformer models on the OLID training data and then combined into an ensemble. 

This repository contains the implementation of our approach proposed in the paper:

> Wiedemann, G., Yimam, S., Biemann, C. (2020): [UHH-LT at SemEval-2020 Task 12: Fine-Tuning of Pre-Trained Transformer Networks for Offensive Language Detection. Proceedings of the Fourteenth Workshop on Semantic Evaluation](https://www.aclweb.org/anthology/2020.semeval-1.213)


Fine-tuning of pre-trained transformer networks such as BERT yield state-of-the-art results for text classification tasks. Typically, fine-tuning is performed on task-specific training datasets in a supervised manner. One can also fine-tune in unsupervised manner beforehand by further pre-training the masked language modeling (MLM) task. Hereby, in-domain data for unsupervised MLM resembling the actual classification target dataset allows for domain adaptation of the model. In this paper, we compare current pre-trained transformer networks with and without MLM fine-tuning on their performance for offensive language detection. Our MLM fine-tuned RoBERTa-based classifier officially ranks 1st in the SemEval 2020 Shared Task 12 for the English language. Further experiments with the ALBERT model even surpass this result.
