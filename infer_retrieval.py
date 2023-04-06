import os
import time
import datetime
import argparse
import logging
import configs

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from zeronlg import ZeroNLG, CaptionDatasetForRetrieval, RetrievalEvaluator
from zeronlg.utils import get_formatted_string

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

try:
    ROOT = configs.annotation_retrieval_root
except:
    ROOT = configs.annotation_root

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--clip_model_name', type=str)
    # Data paths and attributes
    parser.add_argument('--data_root', type=str, default=ROOT)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--val_file', type=str, help='If not specified, use val_file_format')
    parser.add_argument('--test_file', type=str, help='If not specified, use test_file_format')
    parser.add_argument('--pickle_path', type=str, help='If not specified, use pickle_path_format')
    parser.add_argument('--val_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/val.json'))
    parser.add_argument('--test_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/test.json'))
    parser.add_argument('--pickle_path_format', type=str, default=os.path.join(ROOT, '{dataset}/{clip_model_name}_{mode}.pkl'))
    
    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=64)
    
    # Evaluation settings
    parser.add_argument('--modes', type=str, nargs='+', default=['test'], help='evaluation modes: ["val"], ["test"], ["val", "test"]')
    parser.add_argument('--lang', type=str, default='en', help='which language to be generated?')
    parser.add_argument('--num_frames', type=int, default=configs.num_frames)
    parser.add_argument('--mean_pooling', action='store_true')

    # Output settings
    parser.add_argument('--output_path', type=str, help='If not specified, output_path will be {model}/evaluations_retrieval/{dataset}/{lang}')
    parser.add_argument('--no_suffix_folder', action='store_true', help='If True, the suffix `evaluations_retrieval/{dataset}/{lang}` will not be joined to the output path')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        assert args.output_path, "you are training to load a model from hugginface hub, please specify --output_path"

    output_path = args.output_path or args.model
    if not args.no_suffix_folder: 
        output_path = os.path.join(output_path, 'evaluations_retrieval', args.dataset, args.lang)

    os.makedirs(output_path, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(output_path, 'log.txt'), 'w', encoding='utf-8'))
    logger.info(f'output path: {output_path}')

    assert args.modes in [['val'], ['test'], ['val', 'test']]

    logger.info(f'Creating model from {args.model}')
    model = ZeroNLG(args.model, args.clip_model_name)

    # start evaluation
    start_time = time.time()
    for mode in args.modes:
        ann_rpath = get_formatted_string(vars(args), f"{mode}_file", assigned_keys=['dataset', 'lang'])
        logger.info(f'Load dataset from {ann_rpath}')

        pickle_path = get_formatted_string(vars(args), 'pickle_path', assigned_kwargs=dict(
            dataset=args.dataset, clip_model_name=model.clip_model_name, mode=mode,
        ))
        
        dataset = CaptionDatasetForRetrieval(
            vision_root=configs.image_video_root[args.dataset],
            ann_rpath=ann_rpath,
            num_frames=args.num_frames,
            lang=args.lang,
            clip_model=model.clip_model,
            pickle_path=pickle_path,
            logger=logger,
            mean_pooling=args.mean_pooling,
        )
        logger.info(f'There are {len(dataset)} vision inputs')

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        evaluator = RetrievalEvaluator(
            loader=loader,
            mode=mode,
            logger=logger,
            # for MS-COCO dataset, we additionally run 1K test
            # the original 5K test will be also run
            n_fold=5 if args.dataset == 'coco' else 1, 
        )

        evaluator(model, output_path=output_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Time {}'.format(total_time_str))

'''
python infer_retrieval.py --model zeronlg-4langs-vc --output_path output/zeronlg-4langs-vc --dataset msrvtt --lang en --modes val test
'''
