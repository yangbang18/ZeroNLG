import logging
from torch.utils.data import Dataset
from typing import Optional, List
from sentence_transformers import LoggingHandler


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
global_logger = logging.getLogger(__name__)


class TranslateDataset(Dataset):
    def __init__(self, 
                 source_language: str, 
                 target_language: str,
                 source_path: Optional[str] = None,
                 target_path: Optional[str] = None,
                 source_sentences: Optional[List[str]] = None, 
                 target_sentences: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None
                 ) -> None:
        
        self.logger = logger or global_logger
        
        assert source_path is not None or source_sentences is not None
        assert target_path is not None or target_sentences is not None

        if source_sentences is None:
            self.log(f'Loading source sentences ({source_language}) from {source_path}')
            source_sentences = open(source_path, 'r', encoding='utf8').read().strip().split('\n')
        
        if target_sentences is None:
            self.log(f'Loading target sentences ({target_language}) from {target_path}')
            target_sentences = open(target_path, 'r', encoding='utf8').read().strip().split('\n')
        
        assert len(source_sentences) == len(target_sentences), \
            f"#source sents: {len(source_sentences)}; #target sents: {target_sentences}"
        
        self.source_language = source_language
        self.source_sentences = source_sentences
        self.target_language = target_language
        self.target_sentences = target_sentences
    
    def log(self, msg):
        self.logger.info(msg)
    
    def __len__(self):
        return len(self.source_sentences)
    
    def __getitem__(self, index):
        return self.source_sentences[index], self.target_sentences[index]
