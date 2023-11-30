
import os
import time
import datetime
import argparse
import logging

import configs

from torch.utils.data import DataLoader
from zeronlg import ZeroNLG, TranslateDataset, TranslateEvaluator, LoggingHandler


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

try:
    ROOT = configs.annotation_translate_root
except:
    ROOT = configs.annotation_root

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='flickr30k')
    
    # Data paths and attributes
    parser.add_argument('--data_root', type=str, help='If not specified, default to {ROOT}/{dataset}')
    parser.add_argument('--folder_format', type=str, default='{source}-{target}')
    parser.add_argument('--image_list_format', type=str, default='{mode}_images.txt',
        help='To run multimodal machine translation, you should specify a file that stores relative path of images'
    )
    
    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=16)
    
    # Evaluation settings
    parser.add_argument('--source', type=str, default='en', help='source language')
    parser.add_argument('--target', type=str, default='zh', help='target language')
    parser.add_argument('--unidirectional', action='store_true', help='if specified, only evaluating source -> target')
    parser.add_argument('--no_score', action='store_true', help='do not calculate scores')
    parser.add_argument('--modes', type=str, nargs='+', default=['test'], help='evaluation modes: ["val"], ["test"], ["val", "test"]')
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--min_length', type=int, default=5)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    # Output settings
    parser.add_argument('--output_path', type=str, help='If not specified, output_path will be ${model}/evaluations_translate/${source}-${target}')
    parser.add_argument('--no_suffix_folder', action='store_true', help='If True, the suffix `evaluations_translate/${source}-${target}` will not be joined to the output path')
    parser.add_argument('--print_sent', action='store_true')
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(ROOT, args.dataset)
    read_path1 = os.path.join(data_root, args.folder_format.format(source=args.source, target=args.target))
    read_path2 = os.path.join(data_root, args.folder_format.format(source=args.target, target=args.source))
    assert os.path.exists(read_path1) or os.path.exists(read_path2), f'{read_path1} or {read_path2} do not exist!'
    read_path = read_path1 if os.path.exists(read_path1) else read_path2

    if not os.path.exists(args.model):
        assert args.output_path, "you are training to load a model from hugginface hub, please specify --output_path"

    output_path = args.output_path or args.model
    if not args.no_suffix_folder: 
        output_path = os.path.join(output_path, 'evaluations_translate', f'{args.source}-{args.target}')

    os.makedirs(output_path, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(output_path, 'log.txt'), 'w', encoding='utf-8'))
    logger.info(f'output path: {output_path}')

    assert args.modes in [['val'], ['test'], ['val', 'test']]

    logger.info(f'Creating model from {args.model}')
    model = ZeroNLG(args.model, load_clip_model=False)

    # prepare evaluation settings
    evaluation_settings = {
        k: getattr(args, k) 
        for k in ['num_beams', 'max_length', 'min_length', 'repetition_penalty']
    }
    logger.info(f'Evaluation settings: {evaluation_settings}')

    # start evaluation
    start_time = time.time()
    for mode in args.modes:
        if args.unidirectional:
            sources = [args.source]
            targets = [args.target]
        else:
            sources = [args.source, args.target]
            targets = [args.target, args.source]

        for source, target in zip(sources, targets):

            source_path = os.path.join(read_path, f'{mode}.{source}')
            target_path = os.path.join(read_path, f'{mode}.{target}')

            dataset = TranslateDataset(
                source_language=source,
                target_language=target,
                source_path=source_path,
                target_path=target_path,
                logger=logger
            )

            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            
            evaluator = TranslateEvaluator(
                loader=loader,
                evaluation_settings=evaluation_settings,
                mode=mode, 
                logger=logger
            )

            evaluator(model, output_path=output_path, no_score=args.no_score, print_sent=args.print_sent)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Time {}'.format(total_time_str))

'''
python infer_translate.py --model zeronlg-4langs-mt --output_path output/zeronlg-4langs-mt --dataset flickr30k --source zh --target de
'''
