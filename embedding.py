import flair, torch

from flair.embeddings import RoBERTaEmbeddings, DocumentEmbeddings, WordEmbeddings
from flair.data import Sentence, Token
from transformers import AutoTokenizer, AutoConfig, AutoModel
from typing import List, Dict, Set, Tuple

from pathlib import Path
from basetypeutils import AttributeHolder
from abc import abstractmethod

# module to hold the embedding model
class TransformerDocumentEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        model: str = "bert-base-uncased",
        fine_tune: bool = True,
        batch_size: int = 1,
        layers: str = "-1",
        use_scalar_mix: bool = False,
    ):
        """
        Bidirectional transformer embeddings of words from various transformer architectures.
        :param model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for
        options)
        :param fine_tune: If True, allows transformers to be fine-tuned during training
        :param batch_size: How many sentence to push through transformer at once. Set to 1 by default since transformer
        models tend to be huge.
        :param layers: string indicating which layers to take for embedding (-1 is topmost layer)
        :param use_scalar_mix: If True, uses a scalar mix of layers as embedding
        """
        super().__init__()

        # load tokenizer and transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model, config=config)

        # model name
        self.name = str(model)

        # when initializing, embeddings are in eval mode by default
        self.model.eval()
        self.model.to(flair.device)

        # embedding parameters
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.use_scalar_mix = use_scalar_mix
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune
        self.batch_size = batch_size

        # most models have CLS token as last token (GPT-1, GPT-2, TransfoXL, XLNet, XLM), but BERT is initial
        self.initial_cls_token: bool = False
        
        # W use BertTokenizer, so CLS is initial token ...
        self.initial_cls_token = True

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences."""

        # using list comprehension
        sentence_batches = [sentences[i * self.batch_size:(i + 1) * self.batch_size]
                            for i in range((len(sentences) + self.batch_size - 1) // self.batch_size)]

        for batch in sentence_batches:
            self._add_embeddings_to_sentences(batch)

        return sentences

    def _add_embeddings_to_sentences(self, sentences: List[Sentence]):
        """Extract sentence embedding from CLS token or similar and add to Sentence object."""

        # gradients are enabled if fine-tuning is enabled
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with gradient_context:

            # first, subtokenize each sentence and find out into how many subtokens each token was divided
            subtokenized_sentences = []

            # subtokenize sentences
            for sentence in sentences:
                # tokenize and truncate to 512 subtokens (TODO: check better truncation strategies)
                subtokenized_sentence = self.tokenizer.encode(sentence.to_tokenized_string(),
                                                              add_special_tokens=True,
                                                              max_length=512)
                subtokenized_sentences.append(
                    torch.tensor(subtokenized_sentence, dtype=torch.long, device=flair.device))

            # find longest sentence in batch
            longest_sequence_in_batch: int = len(max(subtokenized_sentences, key=len))

            # initialize batch tensors and mask
            input_ids = torch.zeros(
                [len(sentences), longest_sequence_in_batch],
                dtype=torch.long,
                device=flair.device,
            )
            mask = torch.zeros(
                [len(sentences), longest_sequence_in_batch],
                dtype=torch.long,
                device=flair.device,
            )
            for s_id, sentence in enumerate(subtokenized_sentences):
                sequence_length = len(sentence)
                input_ids[s_id][:sequence_length] = sentence
                mask[s_id][:sequence_length] = torch.ones(sequence_length)

            # put encoded batch through transformer model to get all hidden states of all encoder layers
            hidden_states = self.model(input_ids, attention_mask=mask)[-1] if len(sentences) > 1 \
                else self.model(input_ids)[-1]

            # iterate over all subtokenized sentences
            for sentence_idx, (sentence, subtokens) in enumerate(zip(sentences, subtokenized_sentences)):

                index_of_CLS_token = 0 if self.initial_cls_token else len(subtokens) -1

                cls_embeddings_all_layers: List[torch.FloatTensor] = \
                    [hidden_states[layer][sentence_idx][index_of_CLS_token] for layer in self.layer_indexes]

                # use scalar mix of embeddings if so selected
                if self.use_scalar_mix:
                    sm = ScalarMix(mixture_size=len(cls_embeddings_all_layers))
                    sm_embeddings = sm(cls_embeddings_all_layers)

                    cls_embeddings_all_layers = [sm_embeddings]

                # set the extracted embedding for the token
                sentence.set_embedding(self.name, torch.cat(cls_embeddings_all_layers))

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return (
            len(self.layer_indexes) * self.model.config.hidden_size
            if not self.use_scalar_mix
            else self.model.config.hidden_size
        )

    
# ------------------------------------------
# m_name = "roberta-ft-OE2020"
m_name = "roberta-large"
m_layers = "-1,-2,-3,-4"
n_layer = 4
n_dim = 1024

embedding_model = (
    RoBERTaEmbeddings(pretrained_model_name_or_path=m_name, layers=m_layers, pooling_operation="mean"),
    n_layer,
    n_dim
)

cls_embedding_model = TransformerDocumentEmbeddings(
    model = m_name,
    layers = m_layers,
    fine_tune = False,
    batch_size = 8,
    use_scalar_mix = False
)

# embedding_model = RoBERTaEmbeddings(pretrained_model_name_or_path="roberta-base", layers="-1,-2,-3,-4", pooling_operation="mean")
# pooling_operation for subtokens: {first, last, first_last, mean}
# layers are concatenated -> embedding_dim == 3072 (4x768)


# offer backwards compatibility without reloading
loaded_models = { 'RoBERTaEmbeddings' : embedding_model[0] }

def get_model_instance(modelname, loader):
  if modelname not in loaded_models:
    print(f'loading {modelname}')
    loaded_models[modelname] = loader()
  print(f'loaded {modelname}')
  return loaded_models[modelname]

def asTriple(m):
  return (get_model_instance(m.name, m.loadmodel), m.n_layer, m.n_dim)

roberta = AttributeHolder(
    name = 'RoBERTaEmbeddings',
    loadmodel = lambda: RoBERTaEmbeddings(pretrained_model_name_or_path='roberta-base', layers='-1,-2,-3,-4', pooling_operation='mean'),
    n_layer = 4,
    n_dim = 768,
)

glove = AttributeHolder(
    name = 'gloVe',
    loadmodel = lambda: WordEmbeddings(embeddings='en-glove'),
    n_layer = 1,
    n_dim = 300,
)

word2vec_sgns = AttributeHolder(
    name = 'SGNS',
    loadmodel = lambda: WordEmbeddings(embeddings=str(Path('~/data/w2v/GoogleNews-vectors-negative300.bin').expanduser().resolve())),
    n_layer = 1,
    n_dim = 300
)

