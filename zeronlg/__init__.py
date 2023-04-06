__LIBRARY_NAME__ = 'zeronlg'
__version__ = "1.0.0"
__HUGGINGFACE_HUB_NAME__ = 'yangbang18'


from .Framework import Framework
from .ZeroNLG import ZeroNLG
from .losses import LossManager
from .datasets import (
    PretrainDataset, 
    CaptionDataset, 
    CaptionDatasetForRetrieval,
    TranslateDataset
)
from .evaluation import (
    CaptionEvaluator, 
    TranslateEvaluator, 
    RetrievalEvaluator
)
