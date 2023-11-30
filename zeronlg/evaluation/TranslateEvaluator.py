import os
import json
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any, Union, Callable
from sentence_transformers import LoggingHandler
from ..datasets import TranslateDataset
from ..utils import translate_eval
from .. import Framework, ZeroNLG


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
global_logger = logging.getLogger(__name__)


class TranslateEvaluator:
    def __init__(self, 
            loader: DataLoader, 
            evaluation_settings: Dict[str, Any] = {'lang': 'en'}, 
            mode: str = 'val', 
            logger: logging.Logger = None, 
            monitor: str = 'BLEU', 
            with_epoch: bool = False, 
            with_steps: bool = False,
            framework_cls: Callable = Framework,
            seq2seq_cls: Callable = ZeroNLG,
        ):
        super().__init__()
        assert mode in ['val', 'test']
        assert isinstance(loader.dataset, TranslateDataset)

        self.loader = loader
        self.evaluation_settings = evaluation_settings
        self.mode = mode
        self.logger = logger or global_logger
        self.monitor = monitor
        self.with_epoch = with_epoch
        self.with_steps = with_steps
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

        source = self.loader.dataset.source_language
        target = self.loader.dataset.target_language
        self.evaluation_settings['lang'] = target

        if output_path:
            result_file = os.path.join(output_path, f'{prefix}_translations_{source}-{target}.json')
            scores_file = os.path.join(output_path, f'{prefix}_scores_{source}-{target}.json')

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

        results, gts = [], []
        for batch in tqdm(self.loader):
            source_sentences, target_sentences = batch

            outputs = model(
                texts=source_sentences,
                task='translate',
                **self.evaluation_settings,
            )

            results.extend(outputs)
            gts.extend(target_sentences)

            if print_sent:
                for src, trg, pred in zip(source_sentences, target_sentences, outputs):
                    print(f'[SRC] {src}; [TRG] {trg}; [PRED] {pred}')
        
        if output_path:
            self.log(f'Save translation results to {result_file}')
            json.dump(results, open(result_file, 'w'))

        if not no_score:
            scores = translate_eval(gts=gts, res=results, eval_lang=target)
            
            if output_path:
                self.log(f'Save scores to {scores_file}')
                json.dump(scores, open(scores_file, 'w'))

            for k, v in scores.items():
                self.log(f'[{prefix}] [{source} -> {target}] {k} {v}')
            
            score = scores[self.monitor]
            return score
