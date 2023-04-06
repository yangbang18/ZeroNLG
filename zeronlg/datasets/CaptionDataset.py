import os
import logging
import torch
import numpy as np
import json
import pickle
import decord

from sentence_transformers import LoggingHandler
from torch.utils.data import Dataset
from PIL import Image
from .. import Framework
from ..utils import get_uniform_frame_ids


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
global_logger = logging.getLogger(__name__)


class CaptionDataset(Dataset):
    def __init__(self, 
                 vision_root: str, 
                 ann_rpath: str, 
                 num_frames: int = 8, 
                 lang: str = None, 
                 clip_model: Framework = None, 
                 pickle_path: str = None, 
                 logger: logging.Logger = None, 
                 return_images: bool = False, 
                 mean_pooling: bool = False,
                 ):
        
        if return_images:
            assert pickle_path is None
        else:
            assert clip_model is not None

        self.vision_root = vision_root
        self.lang = lang
        self.num_frames = num_frames
        self.clip_model = clip_model
        self.logger = logger or global_logger
        self.return_images = return_images
        self.mean_pooling = mean_pooling

        self.annotation = json.load(open(ann_rpath, 'r'))
        assert 'image' in self.annotation[0], f'{self.annotation[0]} does not contain the key `image`'
        
        self.pickle_path = pickle_path
        self.has_been_updated = False
        if pickle_path is not None and os.path.exists(pickle_path):
            self.log(f'Load CLIP embs from {pickle_path}')
            self.rpath2emb = pickle.load(open(pickle_path, 'rb'))
        else:
            self.log(f'CLIP embs does not exist: {pickle_path}')
            self.rpath2emb = {}

        self.rpath2images = {}

    def get_item(self, image_id):
        for index, ann in enumerate(self.annotation):
            if ann['image_id'] == image_id:
                return self.__getitem__(index)

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        out = {}
        ann = self.annotation[index]

        out['image_id'] = ann['image_id']

        if 'caption' in ann:
            out['text'] = ann['caption']

        rpath = ann['image']

        if rpath not in self.rpath2emb:
            self.has_been_updated = True
            try:
                image_path = os.path.join(self.vision_root, rpath)
                image = Image.open(image_path).convert('RGB')
                images = [image]
            except:
                video_path = os.path.join(self.vision_root, rpath)
                reader = decord.VideoReader(video_path)
                images = reader.get_batch(get_uniform_frame_ids(len(reader), self.num_frames)).asnumpy()
                images = [Image.fromarray(image) for image in images]
            
            if self.clip_model is not None:
                output_value = 'token_embeddings' if self.clip_model.get_module_attribute('use_clip_tokens', False) else 'sentence_embedding'
                emb = self.clip_model.encode(images, output_value=output_value, show_progress_bar=False)
                if isinstance(emb, list):
                    emb = torch.stack(emb, dim=0).cpu().numpy()
                out['emb'] = emb
                self.rpath2emb[rpath] = emb

            if self.return_images:
                out['images'] = images
                if self.clip_model is not None:
                    self.rpath2images[rpath] = images
        else:
            out['emb'] = self.rpath2emb[rpath]
            if self.return_images:
                out['images'] = self.rpath2images[rpath]
        
        out['lang'] = self.lang
        return out

    def log(self, msg):
        self.logger.info(msg)
    
    def save_pickle(self):
        if self.has_been_updated and self.pickle_path is not None:
            self.log(f'Save CLIP embs to {self.pickle_path}')
            with open(self.pickle_path, 'wb') as wf:
                pickle.dump(self.rpath2emb, wf)
        
        self.has_been_updated = False
    
    def collate_fn(self, batch):
        out = {}
        for key in batch[0].keys():
            out[key] = [b[key] for b in batch]
        
        image_ids = [b['image_id'] for b in batch]
        if 'emb' in batch[0]:
            embs = torch.FloatTensor(np.array([b['emb'] for b in batch]))
            if self.mean_pooling:
                embs = embs.mean(1, keepdims=True)
            return image_ids, embs
        else:
            images = [b['images'] for b in batch]
            return image_ids, images


class CaptionDatasetForRetrieval(CaptionDataset):
    def __init__(self, 
                vision_root: str, 
                ann_rpath: str, 
                num_frames: int = 8, 
                lang: str = None, 
                clip_model: Framework = None, 
                pickle_path: str = None, 
                logger: logging.Logger = None, 
                return_images: bool = False, 
                mean_pooling: bool = False,
                ):
        super().__init__(
            vision_root=vision_root, 
            ann_rpath=ann_rpath, 
            num_frames=num_frames, 
            lang=lang, 
            clip_model=clip_model, 
            pickle_path=pickle_path, 
            logger=logger, 
            return_images=return_images, 
            mean_pooling=True, # TODO: we now always apply mean pooling
        )
    
    def collate_fn(self, batch):
        out = {}
        for key in batch[0].keys():
            out[key] = [b[key] for b in batch]
        
        image_ids = [b['image_id'] for b in batch]

        texts = []
        for b in batch:
            if isinstance(b['text'], (list, tuple)):
                texts.extend(b['text'])
            else:
                texts.append(b[texts])

        if 'emb' in batch[0]:
            embs = torch.FloatTensor(np.array([b['emb'] for b in batch]))
            if self.mean_pooling:
                embs = embs.mean(1, keepdims=True)
            return image_ids, embs, texts
        else:
            images = [b['images'] for b in batch]
            return image_ids, images, texts
