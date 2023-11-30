import os
import json
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any, Union, Callable
from sentence_transformers import LoggingHandler
from ..datasets import CaptionDataset
from ..utils import coco_caption_eval
from .. import Framework, ZeroNLG


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
global_logger = logging.getLogger(__name__)


class CaptionEvaluator:
    def __init__(self, 
            loader: DataLoader, 
            gt_file_path: str, 
            evaluation_settings: Dict[str, Any] = {'lang': 'en'}, 
            mode: str = 'val', 
            logger: logging.Logger = None, 
            monitor: str = 'CIDEr', 
            with_epoch: bool = False, 
            with_steps: bool = False,
            auto_save: bool = True,
            framework_cls: Callable = Framework,
            seq2seq_cls: Callable = ZeroNLG,
        ):
        super().__init__()
        assert mode in ['val', 'test']
        assert 'lang' in evaluation_settings
        assert isinstance(loader.dataset, CaptionDataset)

        self.loader = loader
        self.gt_file_path = gt_file_path
        self.evaluation_settings = evaluation_settings
        self.mode = mode
        self.logger = logger or global_logger
        self.monitor = monitor
        self.with_epoch = with_epoch
        self.with_steps = with_steps
        self.auto_save = auto_save
        self.framework_cls = framework_cls
        self.seq2seq_cls = seq2seq_cls

    def log(self, msg):
        self.logger.info(msg)
    
    @torch.no_grad()
    def __call__(self, 
            model: Union[nn.Sequential, nn.Module, str], 
            output_path: str = None, 
            epoch: int = -1, 
            steps: int = -1, 
            no_score: bool=False, 
            print_sent: bool=False
        ) -> float:

        prefix = [self.mode]
        if self.with_epoch:
            prefix.append(f'epoch{epoch}')
        if self.with_steps:
            prefix.append(f'steps{steps}')
        prefix = '_'.join(prefix)

        if output_path:
            result_file = os.path.join(output_path, f'{prefix}_captions.json')
            detailed_scores_file = os.path.join(output_path, f'{prefix}_detailed_scores.json')
            scores_file = os.path.join(output_path, f'{prefix}_scores.json')
        
        if type(model) is str:
            model = self.seq2seq_cls(model)
        elif isinstance(model, self.framework_cls):
            model = self.seq2seq_cls(
                multilingual_model=model,
                device=model.device,
                load_clip_model=False,
            )
        else:
            assert isinstance(model, self.seq2seq_cls)

        results = []
        for batch in tqdm(self.loader):
            image_ids, image_embs = batch

            if self.loader.dataset.clip_model is not None:
                images = None
            else:
                images, image_embs = image_embs, None

            outputs = model(
                images=images,
                image_embs=image_embs, 
                task='caption',
                **self.evaluation_settings
            )
            
            for caption, image_id in zip(outputs, image_ids):
                results.append({"image_id": image_id, "caption": caption})
                if print_sent:
                    print(image_id, caption)
        
        if output_path:
            self.log(f'Save caption results to {result_file}')
            json.dump(results, open(result_file, 'w'))
        
        if self.auto_save:
            self.loader.dataset.save_pickle()

        if not no_score:
            coco_test = coco_caption_eval(self.gt_file_path, result_file, eval_lang=self.evaluation_settings['lang'])
            
            if output_path:
                self.log(f'Save detailed scores to {detailed_scores_file}')
                json.dump(coco_test.evalImgs, open(detailed_scores_file, 'w'))

            if output_path:
                self.log(f'Save scores to {scores_file}')
                json.dump(coco_test.eval, open(scores_file, 'w'))

            for k, v in coco_test.eval.items():
                self.log(f'[{prefix}] {k} {v}')

            score = coco_test.eval[self.monitor]
            return score
