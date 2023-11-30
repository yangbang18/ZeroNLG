import os
import torch
import json
import time
import shutil
import logging
import datetime
import numpy as np
import transformers

import stat
import tempfile
import torch
import sentence_transformers

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm.autonotebook import trange
from collections import OrderedDict
from distutils.dir_util import copy_tree
from huggingface_hub import HfApi, HfFolder, Repository
from typing import List, Dict, Tuple, Iterable, Type, Callable, Optional, Union
from sentence_transformers import LoggingHandler
from .models import Transformer, Pooling, Projector, Decoder
from .utils import (
    MetricLogger, 
    random_masking_, 
    get_cache_folder, 
    seed_everything,
    batch_to_device,
    import_from_string,
    download_if_necessary,
)
from . import (
    __LIBRARY_NAME__, 
    __version__, 
    __HUGGINGFACE_HUB_NAME__,
)


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

global_logger = logging.getLogger(__name__)


sbert_mappings = {
    'sentence_transformers.models.Transformer': 'zeronlg.models.Transformer',
    'sentence_transformers.models.Dense': 'zeronlg.models.Dense',
    'sentence_transformers.models.CLIPModel': 'zeronlg.models.CLIPModel',
    'models.Dense': 'zeronlg.models.Dense',
    'models.Projector': 'zeronlg.models.Projector',
    'models.Decoder': 'zeronlg.models.Decoder',
}


class Framework(sentence_transformers.SentenceTransformer, nn.Sequential):
    def __init__(self, 
                 model_name_or_path: Optional[str] = None, 
                 modules: Optional[Iterable[nn.Module]] = None, 
                 device: Optional[str] = None, 
                 cache_folder: Optional[str] = get_cache_folder(), 
                 use_auth_token: Union[bool, str, None] = None,
                 tie_word_embeddings: bool = True,
                 tie_all: bool = True,
                 init_word_embeddings: bool = False,
                 freeze_word_embeddings: bool = False,
                 logger: logging.Logger = None,
                 load_sbert_only: bool = False,
                 add_version: bool = True,
                 ):

        # check if we need to prefix `model_name_or_path` with __HUGGINGFACE_HUB_NAME__
        if model_name_or_path \
            and 'zeronlg' in model_name_or_path.lower() \
            and '/' not in model_name_or_path \
            and not os.path.exists(model_name_or_path):
            model_name_or_path = os.path.join(__HUGGINGFACE_HUB_NAME__, model_name_or_path)

        self.logger = logger or global_logger
        self.load_sbert_only = load_sbert_only

        # ======= SentenceTransformer Initialization =========
        self._model_card_vars = {}
        self._model_card_text = None
        self._model_config = {}

        if model_name_or_path is not None and model_name_or_path != "":
            self.logger.info("Load pretrained SentenceTransformer: {}".format(model_name_or_path))

            #Old models that don't belong to any organization
            basic_transformer_models = ['albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2', 'albert-xlarge-v1', 'albert-xlarge-v2', 'albert-xxlarge-v1', 'albert-xxlarge-v2', 'bert-base-cased-finetuned-mrpc', 'bert-base-cased', 'bert-base-chinese', 'bert-base-german-cased', 'bert-base-german-dbmdz-cased', 'bert-base-german-dbmdz-uncased', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'bert-base-uncased', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking', 'bert-large-cased', 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-uncased-whole-word-masking', 'bert-large-uncased', 'camembert-base', 'ctrl', 'distilbert-base-cased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-german-cased', 'distilbert-base-multilingual-cased', 'distilbert-base-uncased-distilled-squad', 'distilbert-base-uncased-finetuned-sst-2-english', 'distilbert-base-uncased', 'distilgpt2', 'distilroberta-base', 'gpt2-large', 'gpt2-medium', 'gpt2-xl', 'gpt2', 'openai-gpt', 'roberta-base-openai-detector', 'roberta-base', 'roberta-large-mnli', 'roberta-large-openai-detector', 'roberta-large', 't5-11b', 't5-3b', 't5-base', 't5-large', 't5-small', 'transfo-xl-wt103', 'xlm-clm-ende-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280', 'xlm-mlm-17-1280', 'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024', 'xlm-mlm-enro-1024', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024', 'xlm-roberta-base', 'xlm-roberta-large-finetuned-conll02-dutch', 'xlm-roberta-large-finetuned-conll02-spanish', 'xlm-roberta-large-finetuned-conll03-english', 'xlm-roberta-large-finetuned-conll03-german', 'xlm-roberta-large', 'xlnet-base-cased', 'xlnet-large-cased']

            if os.path.exists(model_name_or_path):
                #Load from path
                model_path = model_name_or_path
            else:
                #Not a path, load from hub
                if '\\' in model_name_or_path or model_name_or_path.count('/') > 1:
                    raise ValueError("Path {} not found".format(model_name_or_path))

                if '/' not in model_name_or_path and model_name_or_path.lower() not in basic_transformer_models:
                    # A model from sentence-transformers
                    model_name_or_path = "sentence-transformers/" + model_name_or_path

                model_path = os.path.join(cache_folder, model_name_or_path.replace("/", "_"))
                
                # Yang B. modification: skip if model path is existed
                if not os.path.exists(model_path):
                    if not os.path.exists(os.path.join(model_path, 'modules.json')):
                        # Download from hub with caching
                        download_if_necessary(model_path, cache_folder, use_auth_token, key_files=['modules.json'])

            if os.path.exists(os.path.join(model_path, 'modules.json')):    #Load as SentenceTransformer model
                modules = self._load_sbert_model(model_path)
            else:   #Load with AutoModel
                modules = self._load_auto_model(model_path)

        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        nn.Sequential.__init__(self, modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)
        # ================

        if tie_word_embeddings:
            self._tie_word_embeddings(tie_all, init_word_embeddings)
        
        if freeze_word_embeddings:
            self._freeze_word_embeddings()
        
        if add_version:
            if '__version__' not in self._model_config:
                self._model_config['__version__'] = {
                    'sentence_transformers': sentence_transformers.__version__,
                    'transformers': transformers.__version__,
                    'pytorch': torch.__version__,
                    __LIBRARY_NAME__: __version__,
                }
            elif __LIBRARY_NAME__ not in self._model_config['__version__']:
                self._model_config['__version__'][__LIBRARY_NAME__] = __version__
        
    def _tie_word_embeddings(self, tie_all: bool=True, init_word_embeddings: bool=False):
        encoder_module, decoder_module = None, None
        for module in self.get_modules():
            if isinstance(module, Transformer):
                encoder_module = module
            if isinstance(module, Decoder):
                decoder_module = module
        
        if decoder_module is not None:
            decoder_input_word_embs = decoder_module.auto_model.get_input_embeddings()
            decoder_output_word_embs = decoder_module.auto_model.get_output_embeddings()

            if encoder_module is not None and (tie_all or init_word_embeddings):
                if encoder_module.tokenizer.get_vocab() == decoder_module.tokenizer.get_vocab():
                    encoder_input_word_embs = encoder_module.auto_model.get_input_embeddings()
                    if tie_all:
                        self.logger.info('decoder\'s input and output word embeddings are tied to encoder\'s word embeddings')
                        decoder_module.auto_model._tie_or_clone_weights(decoder_input_word_embs, encoder_input_word_embs)
                        decoder_module.auto_model._tie_or_clone_weights(decoder_output_word_embs, encoder_input_word_embs)
                    else:
                        self.logger.info('decoder\'s input and output word embeddings are tied, and they are initialized by encoder\'s word embeddings')
                        print(encoder_input_word_embs.weight.norm())
                        decoder_input_word_embs.weight.data = encoder_input_word_embs.weight.data.clone()
                        print(decoder_input_word_embs.weight.norm())
                        decoder_module.auto_model._tie_or_clone_weights(decoder_output_word_embs, decoder_input_word_embs)
                        print(decoder_input_word_embs.weight.norm(), decoder_output_word_embs.weight.norm())

                    new_vocab_size = len(encoder_module.tokenizer.get_vocab())
                    if decoder_module.auto_model.config.vocab_size != new_vocab_size:
                        self.logger.info(f'change `vocab_size` of decoder\'s config from {decoder_module.auto_model.config.vocab_size} to {new_vocab_size}')
                        decoder_module.auto_model.config.vocab_size = new_vocab_size
                else:
                    raise ValueError('tokenizers of the encoder and the decoder are not identical, please do not pass `--tie_all` argument')
            else:
                self.logger.info('decoder\'s input and output word embeddings are tied')
                decoder_module.auto_model._tie_or_clone_weights(decoder_input_word_embs, decoder_output_word_embs)

    def _freeze_word_embeddings(self):
        for module in self.get_modules():
            if isinstance(module, (Transformer, Decoder)):
                for embs in [
                        module.auto_model.get_input_embeddings(), 
                        module.auto_model.get_output_embeddings()
                    ]:
                    if embs is not None:
                        for p in embs.parameters():
                            p.requires_grad = False
    
    def _load_sbert_model(self, model_path):
        """
        Loads a full sentence-transformers model
        """
        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = os.path.join(model_path, 'config_sentence_transformers.json')
        if os.path.exists(config_sentence_transformers_json_path):
            with open(config_sentence_transformers_json_path) as fIn:
                self._model_config = json.load(fIn)

            # Yang B. modification: additionally check version of zeronlg
            for package_name, version in zip(
                    ['sentence_transformers', __LIBRARY_NAME__], 
                    [sentence_transformers.__version__, __version__]
                ):
                if '__version__' in self._model_config \
                    and package_name in self._model_config['__version__'] \
                    and self._model_config['__version__'][package_name] > version:
                    self.logger.warning(
                        f"You try to use a {package_name} model that was created with version {self._model_config['__version__'][package_name]}, however, your version is {version}. \
                        This might cause unexpected behavior or errors.\n\n\n")

        # Check if a readme exists
        model_card_path = os.path.join(model_path, 'README.md')
        if os.path.exists(model_card_path):
            try:
                with open(model_card_path, encoding='utf8') as fIn:
                    self._model_card_text = fIn.read()
            except:
                pass

        # Load the modules of sentence transformer
        modules_json_path = os.path.join(model_path, 'modules.json')
        with open(modules_json_path) as fIn:
            modules_config = json.load(fIn)

        modules = OrderedDict()
        for module_config in modules_config:
            # Yang B. modification: apply mappings, make it compatible to new implementations
            mappings = sbert_mappings
            if hasattr(self, 'sbert_mappings'):
                mappings.update(self.sbert_mappings)
            module_type = mappings.get(module_config['type'], module_config['type'])
            module_class = import_from_string(module_type)
            module = module_class.load(os.path.join(model_path, module_config['path']))
            modules[module_config['name']] = module

        return modules

    def _load_auto_model(self, model_name_or_path):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        # Yang B. modification: check if we automatically load non-sbert model
        if self.load_sbert_only:
            raise FileNotFoundError("No sentence-transformers model found with name {}, and you set `load_sbert_only` to True".format(model_name_or_path))

        self.logger.warning("No sentence-transformers model found with name {}. Creating a new one with MEAN pooling.".format(model_name_or_path))
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'mean')
        return [transformer_model, pooling_model]

    def set_module_attribute(self, module_class, key, value):
        for module in self.get_modules():
            if isinstance(module, module_class):
                setattr(module, key, value)
    
    def get_module_attribute(self, key, default_value=None):
        for module in self.get_modules():
            if hasattr(module, key):
                return getattr(module, key)
        return default_value

    def get_modules(self):
        return [self._modules[_] for _ in iter(self._modules)]
    
    def _get_specific_model(self, before=True, instances=(Projector, Decoder), device=None, return_modules_only: bool = False, **kwargs):
        """only keep related modules"""
        modules = self.get_modules()
        idx = 0
        for module in modules:
            if isinstance(module, instances):
                break
            idx += 1

        device = device or self.device

        if before:
            # get modules < idx
            if idx == 0:
                return None  
            if return_modules_only:
                return modules[:idx]
            model = Framework(modules=modules[:idx], device=device, **kwargs)
        else:
            # get modules >= idx
            if idx == len(modules):
                return None
            if return_modules_only:
                return modules[idx:]
            model = Framework(modules=modules[idx:], device=device, **kwargs)

        model.to(device)
        return model

    def get_encoding_model(self, device=None):
        """return a model that contains modules only related to encoding"""
        return self._get_specific_model(before=True, instances=(Projector, Decoder), device=device or self._target_device, tie_word_embeddings=False)
    
    def get_encoding_modules(self) -> List[nn.Module]:
        """return modules only related to encoding"""
        return self._get_specific_model(before=True, instances=(Projector, Decoder), return_modules_only=True)

    def get_decoding_model(self, device=None):
        """return a model that contains modules only related to decoding"""
        return self._get_specific_model(before=False, instances=(Projector, Decoder), device=device or self._target_device, tie_word_embeddings=False)
    
    def get_decoding_modules(self) -> List[nn.Module]:
        """return modules only related to decoding"""
        return self._get_specific_model(before=False, instances=(Projector, Decoder), return_modules_only=True)

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]], langs: Optional[List[str]]=None):
        module = self._first_module()
        if hasattr(module, 'tokenize'):
            return module.tokenize(texts, langs=langs)
        return {}

    def decoder_tokenize(self, texts: List[str], langs: Optional[List[str]]=None):
        module = self._last_module()
        if isinstance(module, Decoder):
            return module.tokenize(texts, langs=langs)
        return {}
    
    @property
    def tokenizer(self):
        """Property to get the tokenizer that is used by this model"""
        module = self._first_module()
        if hasattr(module, 'tokenizer'):
            return module.tokenizer
        elif hasattr(module, 'processor'):
            return module.processor.tokenizer
        return None
    
    @property
    def decoder_tokenizer(self):
        """Property to get the decoder tokenizer that is used by this model"""
        module = self._last_module()
        if isinstance(module, Decoder):
            return module.tokenizer
        return None
    
    @property
    def device(self):
        return self._target_device

    def smart_batching_collate(self, batch):
        """Transforms a batch of InputExample to features requested by this model"""
        texts = []
        labels = []
        langs = []

        for example in batch:
            texts.append(example.trg_text)
            if example.label is not None:
                labels.append(example.label)
            if example.lang:
                langs.append(example.lang)

        labels = torch.tensor(np.array(labels)) if len(labels) else None

        features = {}

        # prepare input_ids, attention_mask, ...
        tokenized_results = self.tokenize(texts)

        # mask tokenized results if specified
        if getattr(self, 'use_masking', False):
            # self.use_masking and self.mask_prob is defined in Framework.fit
            random_masking_(
                tokenizer=self.tokenizer, 
                tokenized_results=tokenized_results, 
                mask_prob=getattr(self, 'mask_prob', 0.15)
            )

        features.update(tokenized_results) 

        # prepare decoder_input_ids, decoder_attention_mask, ...
        features.update(self.decoder_tokenize(texts, langs if len(langs) else None)) 

        features['source_embedding'] = labels # used for decoding (optional)
        
        return features, labels

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: object = None,
            epochs: int = 1,
            steps_per_epoch: int = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params : Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0,
            log_every: int = 500,
            seed: int = 42,
            use_masking: bool = False,
            mask_prob: float = 0.15,
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (zeronlg.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        """
        seed_everything(seed=seed)

        self.use_masking = use_masking
        self.mask_prob = mask_prob

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self.device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self.device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)


        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        train_start_time = time.time()
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            
            training_steps = 0
            metric_logger = MetricLogger(delimiter="  ")
            start_time = time.time()

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self.device) if labels is not None else None
                    features = batch_to_device(features, self.device)

                    if use_amp:
                        with autocast():
                            loss_value, loss_msg_dict = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value, loss_msg_dict = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()
                    
                    metric_logger.update(**loss_msg_dict)

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if log_every > 0 and global_step % log_every == 0:
                    self.log_training_info(metric_logger, epoch, training_steps, steps_per_epoch)

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                    info = f"[BEST] {self.best_score}"
                    self.logger.info(info)

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
            
            metric_logger.synchronize_between_processes()
            info = f"Averaged stats: {metric_logger.global_avg()}"
            self.logger.info(info)
            time_string = 'Train epoch time: ' + str(datetime.timedelta(seconds=int(time.time() - start_time)))
            self.logger.info(time_string)

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

            if checkpoint_path is not None and checkpoint_save_steps is None:
                self._save_checkpoint_epoch(checkpoint_path, checkpoint_save_total_limit, epoch)

        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            self.save(output_path, create_model_card=False)

        if checkpoint_path is not None and checkpoint_save_steps is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
        
        time_string = 'Train time: ' + str(datetime.timedelta(seconds=int(time.time() - train_start_time)))
        self.logger.info(time_string)
    
    def log_training_info(self, 
            metric_logger: MetricLogger, 
            epoch: int, 
            step: int, 
            steps_per_epoch: int,
            delimiter: str = '  ',
            ):
        
        _msg = [
            'Epoch: {epoch} [{step:' + f'{len(str(steps_per_epoch))}' + 'd} / {steps_per_epoch}]',
            '{meters}',
        ]

        if torch.cuda.is_available():
            _msg.append('max mem: {memory:.0f}')
            MB = 1024.0 * 1024.0
            info = delimiter.join(_msg).format(
                epoch=epoch, 
                step=step, 
                steps_per_epoch=steps_per_epoch, 
                meters=str(metric_logger), 
                memory=torch.cuda.max_memory_allocated() / MB
            )
        else:
            info = delimiter.join(_msg).format(
                epoch=epoch, 
                step=step, 
                steps_per_epoch=steps_per_epoch, 
                meters=str(metric_logger)
            )
        
        self.logger.info(info)
    
    def _save_checkpoint_epoch(self, checkpoint_path, checkpoint_save_total_limit, epoch):
        # Store new checkpoint
        self.save(os.path.join(checkpoint_path, str(epoch)), create_model_card=False)

        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(checkpoint_path):
                if subdir.isdigit():
                    old_checkpoints.append({'epoch': int(subdir), 'path': os.path.join(checkpoint_path, subdir)})

            if len(old_checkpoints) > checkpoint_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x['epoch'])
                shutil.rmtree(old_checkpoints[0]['path'])

    @staticmethod
    def load(input_path):
        return Framework(input_path)
    
    def save_to_hub(self,
                repo_name: str,
                private: Optional[bool] = None,
                commit_message: str = "Add new ZeroNLG model.",
                local_model_path: Optional[str] = None,
                exist_ok: bool = False,
                replace_model_card: bool = False,
                train_datasets: Optional[List[str]] = None):
        """
        Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.

        Yang B. modification: 
        1) delete organization to avoid bugs;

        :param repo_name: Repository name for your model in the Hub.
        :param private: Set to true, for hosting a prive model
        :param commit_message: Message to commit while pushing.
        :param local_model_path: Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
        :param exist_ok: If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
        :param replace_model_card: If true, replace an existing model card in the hub with the automatically created model card
        :param train_datasets: Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.
        :return: The url of the commit of your model in the given repository.
        """
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("You must login to the Hugging Face hub on this computer by typing `transformers-cli login`.")

        endpoint = "https://huggingface.co"
        repo_url = HfApi(endpoint=endpoint).create_repo(
                repo_name,
                token=token,
                private=private,
                repo_type=None,
                exist_ok=exist_ok,
            )
        full_model_name = repo_url[len(endpoint)+1:].strip("/")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # First create the repo (and clone its content if it's nonempty).
            self.logger.info("Create repository and clone it if it exists")
            repo = Repository(tmp_dir, clone_from=repo_url)

            # If user provides local files, copy them.
            if local_model_path:
                copy_tree(local_model_path, tmp_dir)
            else:  # Else, save model directly into local repo.
                create_model_card = False # TODO: zeronlg model card
                self.save(tmp_dir, model_name=full_model_name, create_model_card=create_model_card, train_datasets=train_datasets)

            #Find files larger 5M and track with git-lfs
            large_files = []
            for root, dirs, files in os.walk(tmp_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, tmp_dir)

                    if os.path.getsize(file_path) > (5 * 1024 * 1024):
                        large_files.append(rel_path)

            if len(large_files) > 0:
                self.logger.info("Track files with git lfs: {}".format(", ".join(large_files)))
                repo.lfs_track(large_files)

            self.logger.info("Push model to the hub. This might take a while")
            push_return = repo.push_to_hub(commit_message=commit_message)

            def on_rm_error(func, path, exc_info):
                # path contains the path of the file that couldn't be removed
                # let's just assume that it's read-only and unlink it.
                try:
                    os.chmod(path, stat.S_IWRITE)
                    os.unlink(path)
                except:
                    pass

            # Remove .git folder. On Windows, the .git folder might be read-only and cannot be deleted
            # Hence, try to set write permissions on error
            try:
                for f in os.listdir(tmp_dir):
                    shutil.rmtree(os.path.join(tmp_dir, f), onerror=on_rm_error)
            except Exception as e:
                self.logger.warning("Error when deleting temp folder: {}".format(str(e)))
                pass

        return push_return
