import os
import logging
import gzip
import random
import torch
import numpy as np
from typing import Union, List
from torch.utils.data import Dataset
from sentence_transformers import LoggingHandler
from .. import Framework


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

global_logger = logging.getLogger(__name__)


class InputExample:
    """
    Structure for one input example
    """
    def __init__(self, 
                 sid: str = '', 
                 src_text: str = None,
                 trg_text: str = None,  
                 label: Union[int, float, torch.Tensor, np.ndarray] = 0, 
                 lang: str = None, 
                 ):
        """
        :param sid: id for the example
        :param src_text: the source sentence
        :param trg_text: the target sentence
        :param label: the label for the target sentence
        :param lang: the language of the target sentence
        """
        self.sid = sid
        self.src_text = src_text
        self.trg_text = trg_text
        self.label = label
        self.lang = lang

    def __str__(self):
        return "<InputExample> label: {}, src_text: {}, trg_text: {}, lang: {}".format(
            str(self.label), str(self.src_text), str(self.trg_text), str(self.lang))


class PretrainDataset(Dataset):
    """
    Adapted from sentence_transformers.datasets.ParallelSentencesDataset
    (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/datasets/ParallelSentencesDataset.py)

    This dataset reader can be used to read-in parallel sentences, i.e., it reads in a file with tab-seperated sentences with the same
    sentence in different languages. For example, the file can look like this (EN\tDE\tES):
    hello world     hallo welt  hola mundo
    second sentence zweiter satz    segunda oración

    The sentence in the first column will be mapped to a sentence embedding using the given the embedder. For example,
    embedder is a mono-lingual sentence embedding method for English. The sentences in the other languages will also be
    mapped to this English sentence embedding.

    When getting a sample from the dataset, we get one sentence with the according sentence embedding for this sentence.

    teacher_model can be any class that implement an encode function. The encode function gets a list of sentences and
    returns a list of sentence embeddings
    """

    def __init__(self, 
                 teacher_model: Framework, 
                 batch_size: int = 8, 
                 use_embedding_cache: bool = True, 
                 # Yang B. modification: add extra arguments
                 target_languages: List[str]=None, 
                 logger: logging.Logger=None,
                 numpy_path: str = None,
                 ):
        """
        :param teacher_model: Teacher model, that provides the sentence embeddings for the first column in the dataset file
        :param batch_size: The number of sentences used for embedding extraction per iteration
        :param use_embedding_cache: Cache extracted embeddins for speeding up (if the training lasts multiple epochs)

        :param target_languages: Columns that are not satisfied with the specific target languages will be ignored
        :param logger: If not specified, use the global logger
        :param numpy_path: Path to a numpy file that stores sentence embeddings
        """
        self.teacher_model = teacher_model
        self.datasets = []
        self.datasets_iterator = []
        self.datasets_tokenized = []
        self.dataset_indices = []
        self.copy_dataset_indices = []
        self.cache = []
        self.batch_size = batch_size
        self.use_embedding_cache = use_embedding_cache
        self.embedding_cache = {}
        self.source2index = {}
        self.num_sentences = 0

        self.target_languages = target_languages
        self.logger = logger or global_logger
        self.numpy_path = numpy_path
        if self.numpy_path:
            self.logger.info(f'loading embedding cache from {self.numpy_path}')
            self.embedding_cache = np.load(self.numpy_path)
        if target_languages:
            self.logger.info(f'Target languges during training: {str(self.target_languages)}')

    def load_data(self, 
                  filepath: str, 
                  weight: int = 100, 
                  max_sentences: int = None, 
                  max_sentence_length: int = None, 
                  # Yang B. modification: add extra arguments
                  first_line_is_lang: bool = False, 
                  langs: List[str]=None, 
                  exclude_source: bool=False
                  ):
        """
        Reads in a tab-seperated .txt/.csv/.tsv or .gz file. The different columns contain the different translations of the sentence in the first column

        :param filepath: Filepath to the file
        :param weight: If more than one dataset is loaded with load_data: With which frequency should data be sampled from this dataset?
        :param max_sentences: Max number of lines to be read from filepath
        :param max_sentence_length: Skip the example if one of the sentences is has more characters than max_sentence_length
        :param batch_size: Size for encoding parallel sentences

        :param first_line_is_lang: Whether the first line is the header that indicates the languages of each column (default to False)
        :param langs: The specific languages of all columns (default to None)
        :param exclude_source: Whether exclude sentences in the source langugage (i.e., the first column) as targets (default to False)
        :return:
        """
        logger = self.logger

        logger.info("Load "+filepath)
        parallel_sentences = []
        first_line_flag = True

        # Yang B. modification: record parallel languages of this data if specified
        if (first_line_is_lang or langs):
            if not hasattr(self, 'langs_of_data'):
                self.langs_of_data = []
            if langs:
                self.langs_of_data.append(langs)
                logger.info(f"There are {len(langs)} langauges: {langs}")
        elif hasattr(self, 'langs_of_data'):
            # it means that you have a inconsistent behavior when calling this function.
            raise ValueError('You should pass `first_line_is_lang` = True or specify langs for all data')
        
        with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
            count = 0
            for line in fIn:
                sentences = line.strip().split("\t")

                # check languages
                if first_line_flag and first_line_is_lang:
                    first_line_flag = False
                    if langs:
                        assert len(langs) == len(sentences)
                        for lang1, lang2 in zip(langs, sentences):
                            assert lang1 == lang2
                    else:
                        self.langs_of_data.append(sentences)
                        logger.info(f"There are {len(sentences)} langauges: {sentences}")
                    continue
                
                if hasattr(self, 'langs_of_data'):
                    # ensure that each line has the same number of sentences as that of languages
                    assert len(sentences) == len(self.langs_of_data[-1]), f"{sentences}, {len(self.langs_of_data[-1])}"
            
                if max_sentence_length is not None and max_sentence_length > 0 and max([len(sent) for sent in sentences]) > max_sentence_length:
                    continue

                parallel_sentences.append(sentences)
                count += 1
                if max_sentences is not None and max_sentences > 0 and count >= max_sentences:
                    break

        # show statistics and an example
        logger.info(f"There are {count} lines, one of which is {parallel_sentences[0]}")

        self.add_dataset(parallel_sentences, weight, max_sentences, max_sentence_length, exclude_source)
    
    def add_dataset(self, 
                    parallel_sentences: List[List[str]], 
                    weight: int = 100, 
                    max_sentences: int = None, 
                    max_sentence_length: int = 128, 
                    # Yang B. modification: add extra arguments
                    exclude_source: bool = False
                    ):
        sentences_map = {}
        for idx, sentences in enumerate(parallel_sentences):
            if max_sentence_length is not None and max_sentence_length > 0 and max([len(sent) for sent in sentences]) > max_sentence_length:
                continue

            source_sentence = sentences[0]
            
            self.source2index[source_sentence] = idx

            if source_sentence not in sentences_map:
                sentences_map[source_sentence] = set()

            if hasattr(self, 'langs_of_data'):
                langs = self.langs_of_data[-1]
            else:
                langs = [None for _ in range(len(sentences))]

            # whether we exclude the source sentences as a part of targets
            # we carry out this operation to avoid imbalanced language distribution
            # e.g., if we add datasets of columns A-B, A-C, and A-D with `exclude_source = False`
            # then A: B: C: D = 3: 1: 1: 1
            start = 1 if exclude_source else 0
            for i in range(start, len(sentences)):
                sent = sentences[i]
                lang = langs[i]
                if self.target_languages and lang not in self.target_languages:
                    continue
                sentences_map[source_sentence].add((sent, lang))

            if max_sentences is not None and max_sentences > 0 and len(sentences_map) >= max_sentences:
                break

        if len(sentences_map) == 0:
            return

        self.num_sentences += sum([len(sentences_map[sent]) for sent in sentences_map])

        dataset_id = len(self.datasets)
        self.datasets.append(list(sentences_map.items()))
        self.datasets_iterator.append(0)
        self.dataset_indices.extend([dataset_id] * weight)
    
    def generate_data(self):
        # Yang B. modification: add the language of each target sentence (if available) and the source text into the InputExample
        source_sentences_list = []
        target_sentences_list = []
        target_languages_list = []

        for data_idx in self.dataset_indices:
            src_sentence, trg_sentences = self.next_entry(data_idx)
            source_sentences_list.append(src_sentence)
            target_sentences_list.append([item[0] for item in trg_sentences])
            target_languages_list.append([item[1] for item in trg_sentences])

        #Generate embeddings
        src_embeddings = self.get_embeddings(source_sentences_list)

        for src_sentence, src_embedding, trg_sentences, trg_languages, data_idx in zip(
                source_sentences_list, 
                src_embeddings, 
                target_sentences_list, 
                target_languages_list, 
                self.dataset_indices
            ):
            for trg_sentence, trg_language in zip(trg_sentences, trg_languages):
                self.cache.append(
                    InputExample(
                        src_text=src_sentence, 
                        trg_text=trg_sentence, 
                        label=src_embedding, 
                        lang=trg_language,
                    )
                )

        random.shuffle(self.cache)
    
    def next_entry(self, data_idx):
        source, target_sentences = self.datasets[data_idx][self.datasets_iterator[data_idx]]

        self.datasets_iterator[data_idx] += 1
        if self.datasets_iterator[data_idx] >= len(self.datasets[data_idx]): #Restart iterator
            self.datasets_iterator[data_idx] = 0
            random.shuffle(self.datasets[data_idx])

        return source, target_sentences

    def get_embeddings(self, sentences):
        # Yang B. modification: if we have loaded numpy file, directly return embeddings
        if self.numpy_path:
            embeddings = [self.embedding_cache[self.source2index[source]] for source in sentences]
            return embeddings

        if self.teacher_model is None:
            return [None for sent in sentences]

        if not self.use_embedding_cache:
            return self.teacher_model.encode(sentences, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)

        #Use caching
        new_sentences = []
        for sent in sentences:
            if sent not in self.embedding_cache:
                new_sentences.append(sent)

        if len(new_sentences) > 0:
            new_embeddings = self.teacher_model.encode(new_sentences, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
            for sent, embedding in zip(new_sentences, new_embeddings):
                self.embedding_cache[sent] = embedding

        return [self.embedding_cache[sent] for sent in sentences]

    def __len__(self):
        return self.num_sentences

    def __getitem__(self, idx):
        if len(self.cache) == 0:
            self.generate_data()

        return self.cache.pop()
