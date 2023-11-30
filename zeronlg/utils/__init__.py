from .eval import coco_caption_eval, translate_eval
from .metriclogger import MetricLogger
from .op_text import random_masking_
from .op_vision import get_uniform_frame_ids, process_images
from .io import get_cache_folder, get_formatted_string, download_if_necessary

from sentence_transformers.util import (
    batch_to_device,
    fullname, 
    import_from_string,
    snapshot_download
)


def seed_everything(seed=42):
    import torch
    import random
    import numpy as np
    import torch
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
