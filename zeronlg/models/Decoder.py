import os
import json
import torch
import transformers
from torch import nn
from typing import List, Dict, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sentence_transformers.util import snapshot_download
from ..utils import get_cache_folder
from .. import __LIBRARY_NAME__, __version__


# map a language to a specific token, 
# which will be used as the begin-of-sentence (BOS) token to guide the decoder
LANG2TOKEN = {
    'en': '[unused1]',
    'zh': '[unused2]',
    'de': '[unused3]',
    'fr': '[unused4]',
} # Note: do not modify the existing mappings in LANG2TOKEN, you can instead add new ones


class Decoder(nn.Module):
    """Huggingface AutoModelForCausalLM for decoding.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    :param from_pretrained: If true, load pre-trained weights (deafult to true)
    :param attend_to: A list of string(s) to specify which encoder(s) will the decoder attend to,
                      e.g., ['student'] means taking the newly-trained encoder's outputs as the cross-attention inputs;
                            ['teacher'] means taking the (frozen) pre-trained encoder's outputs as the cross-attention inputs;
                            ['student', 'teacher'] ...
    :param teacher_model_name: The name of the teacher model, which will be stored into the module config and used for re-loading
    """
    def __init__(self, 
                 model_name_or_path: str, 
                 model_args: Dict = {}, 
                 max_seq_length: Optional[int] = None,
                 cache_folder: Optional[str] = None,
                 tokenizer_args: Dict = {}, 
                 do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None,
                 from_pretrained: bool = True,
                 use_auth_token: Union[bool, str, None] = None,
                 attend_to: List[str] = ['student'],
                 teacher_model_name: str = None,
                 use_clip_tokens: Optional[bool] = None,
                 ):
        super().__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case', 'attend_to', 'teacher_model_name', 'use_clip_tokens']
        self.do_lower_case = do_lower_case
        self.teacher_model_name = teacher_model_name
        self.use_clip_tokens = bool(use_clip_tokens or False)

        assert isinstance(attend_to, (list, tuple))
        self.attend_to = list(set(attend_to))

        cache_folder = get_cache_folder(cache_folder)
        if os.path.exists(model_name_or_path):
            model_path = model_name_or_path
        else:
            model_path = os.path.join(cache_folder, model_name_or_path.replace('/', '_'))
        
        if not os.path.exists(os.path.join(model_path, 'config.json')):
            storage_path = snapshot_download(model_name_or_path,
                            cache_dir=cache_folder,
                            library_name=__LIBRARY_NAME__,
                            library_version=__version__,
                            ignore_files=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'],
                            use_auth_token=use_auth_token)
            assert model_path == storage_path
        
        assert os.path.exists(model_path)

        config = AutoConfig.from_pretrained(model_path, **model_args)
        assert config.is_decoder is True
        assert config.add_cross_attention is True

        self._load_model(model_path, config, from_pretrained)
        self.auto_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_path, **tokenizer_args)
        self.vocab = self.tokenizer.get_vocab()
        
        self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

        # determine the behavior during generation based on the version of transformers
        self.should_repeat = True
        version = [int(item) for item in transformers.__version__.split('.')]
        if version[0] > 4 or (version[0] == 4 and version[1] >=27):
            # after 4.27.0, we should not repeat encoder's outputs by `num_beams` * `num_return_sequences` times
            self.should_repeat = False

    def _load_model(self, model_name_or_path, config, from_pretrained):
        """Loads the transformer model"""
        if from_pretrained:
            self.auto_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
        else:
            self.auto_model = AutoModelForCausalLM.from_config(config)

    def __repr__(self):
        return "Decoder({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)

    def get_encoder_attention_mask(self, encoder_hidden_states, features, attend_to, n_repeats=1):
        encoder_attention_mask = None
        if encoder_hidden_states.size(1) != 1 and attend_to == 'student':
            encoder_attention_mask = features['attention_mask']
        
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.repeat_interleave(n_repeats, dim=0)
        
        return encoder_attention_mask

    def prepare_encoder_hidden_states(self, states):
        B = states.size(0)
        D = states.size(-1)
        if states.ndim == 2:
            states = states.unsqueeze(1)
        if states.ndim == 4:
            states = states.view(B, -1, D)
        return states

    def forward(self, features):
        """Returns losses if self.training is True, otherwise generation results"""

        attend_to = features.get('attend_to', None) or self.attend_to

        if not isinstance(attend_to, (list, tuple)):
            attend_to = [attend_to]

        if self.training:
            inputs = {
                'input_ids': features['decoder_input_ids'], 
                'attention_mask': features.get('decoder_attention_mask', None),
                'token_type_ids': features.get('decoder_token_type_ids', None),
            }
            inputs['labels'] = inputs['input_ids'].masked_fill(inputs['input_ids'] == self.tokenizer.pad_token_id, -100)

            for at in attend_to:
                assert at in ['teacher', 'student'], f"`attend_to` == {at} is not supported during training yet"

                # if attending to the teacher's hidden states, it denotes a English -> English/Chinese/German/French process
                # if attending to the student's hidden states, it denotes a auto-encoding process
                source = features[f'{at}_hidden_states']
                
                inputs['encoder_hidden_states'] = self.prepare_encoder_hidden_states(source)
                inputs['encoder_attention_mask'] = self.get_encoder_attention_mask(source, features, at)
                
                output_states = self.auto_model(**inputs, return_dict=False)
                loss = output_states[0]

                features.update({f'loss_at_{at}': loss})
        else:
            # if we do not filter those unused keys, there will be an error for transformers==4.27.1
            ignore_keys = ['sentence_embedding', 'source_embedding', 'attend_to', 'decoder_input_ids', 'student_hidden_states', 'teacher_hidden_states', 'token_embeddings', 'num_frames']
            generate_kwargs = {k: v for k, v in features.items() if k not in ignore_keys}

            for at in attend_to:
                assert at in ['student', 'teacher', 'both']
                if at == 'both':
                    # multimodal machine translation
                    encoder_hidden_states = torch.cat([features['teacher_hidden_states'], features['student_hidden_states']], dim=1)
                else:
                    # visual captioning (at == 'teacher') or machine translation (at == 'student')
                    encoder_hidden_states = features[f'{at}_hidden_states']
                encoder_hidden_states = self.prepare_encoder_hidden_states(encoder_hidden_states)

                if self.should_repeat:
                    n_repeats = generate_kwargs['num_beams']
                    if generate_kwargs.get('do_sample', False):
                        n_repeats *= generate_kwargs.get('num_return_sequences', 1)
                    encoder_hidden_states = encoder_hidden_states.repeat_interleave(n_repeats, dim=0)
                else:
                    n_repeats = 1
                encoder_attention_mask = self.get_encoder_attention_mask(encoder_hidden_states, features, at, n_repeats=n_repeats)

                inputs = {
                    'input_ids': features['decoder_input_ids'], 
                    'encoder_hidden_states': encoder_hidden_states,
                    'attention_mask': None,
                    'encoder_attention_mask': encoder_attention_mask,
                    'eos_token_id': self.eos_token_id,
                    'pad_token_id': self.pad_token_id,
                }
                generate_kwargs.update(inputs)

                outputs = self.auto_model.generate(**generate_kwargs)
                features[f'results_at_{at}'] =  self._get_captions(outputs)

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: List[str], langs: Optional[List[str]]=None):
        """Tokenizes texts and maps tokens to token-ids"""
        to_tokenize = texts
        #strip
        to_tokenize = [str(text).strip() for text in to_tokenize]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [text.lower() for text in to_tokenize]

        outputs = self.tokenizer(to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length)

        for input_ids in outputs['input_ids']:
            assert self.bos_token_id in input_ids
            assert self.eos_token_id in input_ids

        if langs:
            assert len(outputs['input_ids']) == len(langs)
            for input_ids, lang in zip(outputs['input_ids'], langs):
                assert lang in LANG2TOKEN, f"{lang} not in LANG2TOKEN {LANG2TOKEN.keys()}"
                
                lang_token_id = self.vocab.get(LANG2TOKEN[lang], None)
                if not lang_token_id:
                    raise NotImplementedError(f'The special token of {lang}, i.e., {LANG2TOKEN[lang]}, is not found in the vocab; You may call tokenizer.add_tokens')
                
                index_of_bos_token_id = input_ids.numpy().tolist().index(self.bos_token_id)
                # override the first bos token with lang token
                input_ids[index_of_bos_token_id] = lang_token_id
                
        return {f'decoder_{k}': v for k, v in outputs.items()}
    
    def get_bos_input_ids(self, batch_size: int, lang: Optional[str]=None):
        bos = self.bos_token_id if lang is None else self.vocab[LANG2TOKEN[lang]]
        return torch.LongTensor([[bos] for _ in range(batch_size)])

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'decoder_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        sbert_config_path = os.path.join(input_path, 'decoder_config.json')

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Decoder(model_name_or_path=input_path, **config)
    
    def _get_captions(self, caption_ids):
        captions = []
        for i, output in enumerate(caption_ids):
            # skip the bos token, which can be not a special token, e.g., [unused1]
            caption = self.tokenizer.decode(output[1:], skip_special_tokens=True)
            captions.append(caption)
        return captions

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        assert kwargs.get('encoder_hidden_states', None) is not None
        
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "past_key_values": past, 
            'encoder_hidden_states': kwargs['encoder_hidden_states'],
            'encoder_attention_mask': kwargs['encoder_attention_mask'],
        }
