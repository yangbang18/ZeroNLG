import torch
import torch.nn.functional as F
from torch import nn, Tensor
from PIL import Image
from typing import List, Optional, Union, Dict, Any, Tuple
from sentence_transformers.util import batch_to_device

from . import Framework
from .utils import process_images
from .models import Decoder, CLIPModel


SUPPORTED_TASKS = ['caption', 'translate']


class ZeroNLG(nn.Module):
    def __init__(self, 
                 multilingual_model: Union[str, Framework], 
                 clip_model: Union[str, Framework, None] = None, 
                 use_clip_tokens: Optional[bool] = None,
                 load_clip_model: bool = True,
                 device: Union[str, torch.device, None] = None,
        ):
        super().__init__()
        self.use_clip_tokens = use_clip_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._target_device = torch.device(device)

        if type(multilingual_model) is str:
            self.multilingual_model = Framework(multilingual_model, device=self.device)
        else:
            self.multilingual_model = multilingual_model

        self.clip_model_name = None
        if clip_model is None or clip_model == '':
            self.clip_model_name = self.multilingual_model.get_module_attribute('teacher_model_name') \
                    or self.multilingual_model.get_module_attribute('clip_model_name')
        elif type(clip_model) is str:
            self.clip_model_name = clip_model
        else:
            assert isinstance(clip_model, Framework)
            self.clip_model = clip_model

        if load_clip_model:
            self._load_clip_model()

    def forward(self,
            images: Union[str, List[str], Image.Image, List[Image.Image], None] = None,
            texts: Union[str, List[str], None] = None, 
            image_embs: Optional[Tensor] = None,
            text_embs: Optional[Tensor] = None,
            num_frames: int=8,
            lang: Optional[str] = None, 
            num_beams: int = 3,
            max_length: int = 30,
            min_length: int = 5,
            repetition_penalty: float = 1.0,
            return_all: bool = False,
            attend_to: Union[str, List[str], None] = None,
            task: Optional[str] = None,
            **kwargs,
        ) -> Union[List[str], Dict[str, Any]]:

        if task:
            assert task in SUPPORTED_TASKS
        else:
            raise ValueError(f'Please either pass the argument `task` ({SUPPORTED_TASKS}) or call the corresponding forward function')

        forward_func_name = f'forward_{task}'
        forward_func = getattr(self, forward_func_name, None)
        assert forward_func is not None, f"Please implement the function {forward_func_name} in ZeroNLG"

        return forward_func(
            images=images,
            texts=texts,
            image_embs=image_embs,
            text_embs=text_embs,
            num_frames=num_frames,
            lang=lang,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            return_all=return_all,
            attend_to=attend_to,
            **kwargs
        )
    
    def forward_caption(self, 
            images: Union[str, List[str], Image.Image, List[Image.Image], None] = None,
            image_embs: Optional[Tensor] = None,
            num_frames: int = 8,
            lang: Optional[str] = None, 
            num_beams: int = 3,
            max_length: int = 30,
            min_length: int = 5,
            repetition_penalty: float = 1.0,
            return_all: bool = False,
            attend_to: Union[str, List[str], None] = None,
            **kwargs
        ) -> Union[List[str], Dict[str, Any]]:

        # the decoder should always attend to teacher's outputs in the cross-attention layers
        attend_to = ['teacher']

        # prepare clip image embeddings
        image_embs = self.get_image_embeddings(images, image_embs, num_frames, normalize=False, mean_pooling=False)

        # we re-define multilingual model to exclude encoding modules, which are useless for visual captioning
        multilingual_model = self.multilingual_model.get_decoding_model(self.device)
        multilingual_model.eval()

        # features that will be passed to the multilingual model
        features = dict(
            source_embedding=image_embs,
            attend_to=attend_to,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
        )
        features.update(kwargs) # kwargs for generation
        features['decoder_input_ids'] = self.get_bos_input_ids(batch_size=image_embs.size(0), lang=lang)
        features = batch_to_device(features, self.device)

        with torch.no_grad():
            outputs = multilingual_model(features)

        if return_all:
            return outputs

        # return only the caption
        return outputs['results_at_teacher']

    def forward_translate(self, 
            texts: Union[str, List[str], Tuple[str], None] = None, 
            text_embs: Optional[Tensor] = None,
            lang: Optional[str] = None, 
            num_beams: int = 3,
            max_length: int = 30,
            min_length: int = 5,
            repetition_penalty: float = 1.0,
            return_all: bool = False,
            attend_to: Union[str, List[str], None] = None,
            **other_generate_kwargs
        ) -> Union[List[str], Dict[str, Any]]:

        # by default, the decoder attend to student's output in cross-attention layers
        attend_to = attend_to or ['student']
        if type(attend_to) is str:
            attend_to = [attend_to]
        if text_embs is not None:
            # given that you have provided clip's text text_embs, the decoder must attend to clip (teacher)
            attend_to = ['teacher']
        assert len(attend_to) == 1, 'attend_to should be one of "teacher", "student", ["teacher"], ["student"]'
        
        assert texts is not None or text_embs is not None, "you should specify either texts or text_embs"

        # prepare tokenized text features for the multilingual encoder
        # or clip text embeddings
        tokenized_features, source_embedding = {}, None
        if text_embs is None:
            texts = [texts] if not isinstance(texts, (list, tuple)) else texts
            batch_size = len(texts)

            if attend_to == ['teacher']:
                # extract clip text embeddigns
                self._load_clip_model() # load clip model if it has not been loaded
                self.clip_model = self.clip_model.to(self.device)
                self.clip_model.eval()
                source_embedding = self.clip_model.encode(texts, batch_size, show_progress_bar=False, convert_to_tensor=True)
            else:
                tokenized_features = self.multilingual_model.tokenize(texts)
        else:            
            source_embedding = text_embs
            batch_size = text_embs.size(0)

        # re-define multilingual model if necessary
        if attend_to == ['teacher']:
            multilingual_model = self.multilingual_model.get_decoding_model(self.device)
        else:
            multilingual_model = self.multilingual_model.to(self.device)
        multilingual_model.eval()

        # features that will be passed to the multilingual model
        features = dict(
            source_embedding=source_embedding,
            attend_to=attend_to,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            **tokenized_features,
        )
        features.update(other_generate_kwargs)
        features['decoder_input_ids'] = self.get_bos_input_ids(batch_size=batch_size, lang=lang)
        features = batch_to_device(features, self.device)

        with torch.no_grad():
            outputs = multilingual_model(features)

        if return_all:
            return outputs

        # return only the translated text
        return outputs[f'results_at_{attend_to[0]}']

    def get_image_embeddings(self,
            images: Union[str, List[str], Image.Image, List[Image.Image], None] = None,
            image_embs: Optional[Tensor] = None,
            num_frames: int = 8,
            mean_pooling: bool = False,
            normalize: bool = False,
            batch_size: Optional[int] = None,
            **kwargs,
        ) -> Tensor:
        """Extract CLIP image embeddings"""

        if image_embs is None:
            assert images is not None, "you should specify either images or image_embs"
            
            self._load_clip_model() # load clip model if it has not been loaded
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()

            images, is_video, num_frames, num_samples = process_images(images, num_frames)
            
            batch_size = batch_size or num_samples
            
            image_embs = self.clip_model.encode(
                images, batch_size, output_value='token_embeddings' if self.use_clip_tokens else 'sentence_embedding',
                show_progress_bar=False, convert_to_tensor=True, device=self.device)
            
            if isinstance(image_embs, list):
                image_embs = torch.stack(image_embs, dim=0).to(self.device)

            if is_video:
                image_embs = image_embs.view(batch_size, num_frames, -1, image_embs.size(-1)).squeeze(2)
        else:
            image_embs = image_embs.to(self.device)
            batch_size = image_embs.size(0)

        if image_embs.ndim == 1:
            image_embs = image_embs.unsqueeze(0)
        
        if image_embs.ndim > 2 and mean_pooling:
            # averaged over the time axis
            image_embs = image_embs.mean(dim=1)
        
        if normalize:
            image_embs = F.normalize(image_embs, dim=-1)
        
        return image_embs
    
    def get_text_embeddings(self, 
            texts: Union[str, List[str], None] = None, 
            text_embs: Optional[Tensor] = None,
            normalize: bool = False,
            batch_size: Optional[int] = None,
            **kwargs,
        ) -> Tensor:
        """Extract CLIP text embeddings"""

        # we re-define multilingual model to exclude useless decoding modules
        multilingual_model = self.multilingual_model.get_encoding_model(self.device)
        multilingual_model.eval()

        texts = [texts] if not isinstance(texts, (list, tuple)) else texts
        batch_size = batch_size or len(texts)

        if text_embs is None:
            text_embs = multilingual_model.encode(texts, batch_size, show_progress_bar=False, convert_to_tensor=True, device=self.device)
        else:
            text_embs = text_embs.to(self.device)
        
        if text_embs.ndim == 1:
            text_embs = text_embs.unsqueeze(0)
        
        if normalize:
            text_embs = F.normalize(text_embs, dim=-1)

        return text_embs

    def get_bos_input_ids(self, batch_size: int, lang: Optional[str] = None) -> Tensor:
        for module in self.multilingual_model.get_modules():
            if isinstance(module, Decoder):
                return module.get_bos_input_ids(batch_size=batch_size, lang=lang)
    
    def _load_clip_model(self):
        if not hasattr(self, 'clip_model'):
            try:
                # in this case, the multilignual model is actually a monolingual CLIP model
                print(self.multilingual_model)
                assert isinstance(self.multilingual_model._first_module(), CLIPModel)
                self.clip_model = self.multilingual_model.get_encoding_model(device=self.device)
            except:
                assert self.clip_model_name is not None, "you are trying to use a clip model, whose name can not be obtained from the multilingual model;\
                    Maybe you should pass the argument `clip_model_name` when defining a ZeroNLG model"
                assert type(self.clip_model_name) is str
                self.clip_model = Framework(self.clip_model_name, device=self.device)
            
            self.use_clip_tokens = self.use_clip_tokens or self.multilingual_model.get_module_attribute('use_clip_tokens', False)
            self.clip_model.set_module_attribute(CLIPModel, 'use_clip_tokens', self.use_clip_tokens)

    @property
    def device(self):
        return self._target_device
