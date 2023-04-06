import os
import json
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from typing import Dict, Any, Union

from .. import Framework, ZeroNLG
from ..datasets import TranslateDataset
from ..utils import translate_eval


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

    def log(self, msg):
        self.logger.info(msg)
    
    @torch.no_grad()
    def __call__(self, 
            model: Union[Framework, ZeroNLG], 
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

        if isinstance(model, Framework):
            zeronlg = ZeroNLG(
                multilingual_model=model,
                device=model.device,
                load_clip_model=False,
            )
        else:
            assert isinstance(model, ZeroNLG)
            zeronlg = model

        results, gts = [], []
        for batch in tqdm(self.loader):
            source_sentences, target_sentences = batch

            outputs = zeronlg.forward_translate(
                texts=source_sentences,
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
