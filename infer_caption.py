
import os
import sys
import time
import datetime
import argparse
import logging
import configs

from torch.utils.data import DataLoader
from zeronlg import ZeroNLG, CaptionDataset, CaptionEvaluator, LoggingHandler
from zeronlg.utils import get_formatted_string, coco_caption_eval


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

try:
    ROOT = configs.annotation_caption_root
except:
    ROOT = configs.annotation_root

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--use_clip_tokens', type=int, help='Whether use token-level visual embeddings?')
    # Data paths and attributes
    parser.add_argument('--data_root', type=str, default=ROOT)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--val_file', type=str, help='If not specified, use val_file_format')
    parser.add_argument('--val_gt_file', type=str, help='If not specified, use val_gt_file_format')
    parser.add_argument('--test_file', type=str, help='If not specified, use test_file_format')
    parser.add_argument('--test_gt_file', type=str, help='If not specified, use test_gt_file_format')
    parser.add_argument('--pickle_path', type=str, help='If not specified, use pickle_path_format')
    parser.add_argument('--val_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/val.json'))
    parser.add_argument('--val_gt_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/val_gt.json'))
    parser.add_argument('--test_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/test.json'))
    parser.add_argument('--test_gt_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/test_gt.json'))
    parser.add_argument('--pickle_path_format', type=str, default=os.path.join(ROOT, '{dataset}/{clip_model_name}_{mode}{postfix}.pkl'))
    
    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=64)
    
    # Evaluation settings
    parser.add_argument('--auto', action='store_true', help='whether to use the auto_settings')
    parser.add_argument('--no_score', action='store_true', help='do not calculate caption scores')
    parser.add_argument('--modes', type=str, nargs='+', default=['test'], help='evaluation modes: ["val"], ["test"], ["val", "test"]')
    parser.add_argument('--lang', type=str, default='en', help='which language to be generated?')
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=30)
    parser.add_argument('--min_length', type=int, default=5)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--num_frames', type=int, default=configs.num_frames)
    parser.add_argument('--mean_pooling', action='store_true')

    # Output settings
    parser.add_argument('--output_path', type=str, help='If not specified, output_path will be {model}/evaluations_caption/{dataset}/{lang}')
    parser.add_argument('--no_suffix_folder', action='store_true', help='If True, the suffix `evaluations_caption/{dataset}/{lang}` will not be joined to the output path')
    parser.add_argument('--print_sent', action='store_true')
    args = parser.parse_args()

    if args.results_file:
        logger.info(f'results_file: {args.results_file}')
        for mode in args.modes:
            assert mode in ['val', 'test']
            gt_file = f'{ROOT}/{args.dataset}/{args.lang}/{mode}_gt.json'
            logger.info(f'gt_file: {gt_file}')
            coco_caption_eval(gt_file, args.results_file, eval_lang=args.lang)
        sys.exit(0)
    else:
        assert args.model is not None, "please specify --model"

    if not os.path.exists(args.model):
        assert args.output_path, "you are training to load a model from hugginface hub, please specify --output_path"

    output_path = args.output_path or args.model
    if not args.no_suffix_folder: 
        output_path = os.path.join(output_path, 'evaluations_caption', args.dataset, args.lang)

    os.makedirs(output_path, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(output_path, 'log.txt'), 'w', encoding='utf-8'))
    logger.info(f'output path: {output_path}')

    assert args.modes in [['val'], ['test'], ['val', 'test']]

    logger.info(f'Creating model from {args.model}')
    model = ZeroNLG(args.model, use_clip_tokens=args.use_clip_tokens)

    # prepare evaluation settings
    evaluation_settings = {
        k: getattr(args, k) 
        for k in ['lang', 'num_beams', 'max_length', 'min_length', 'repetition_penalty']
    }
    if args.auto:
        evaluation_settings.update(configs.auto_settings[args.lang])
    logger.info(f'Evaluation settings: {evaluation_settings}')

    # start evaluation
    start_time = time.time()
    for mode in args.modes:
        ann_rpath = get_formatted_string(vars(args), f"{mode}_file", assigned_keys=['dataset', 'lang'])
        logger.info(f'Load dataset from {ann_rpath}')

        pickle_path = get_formatted_string(vars(args), 'pickle_path', assigned_kwargs=dict(
            dataset=args.dataset, clip_model_name=model.clip_model_name, mode=mode, postfix='_tokens' if model.use_clip_tokens else ''
        ))

        dataset = CaptionDataset(
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

        gt_file_path = get_formatted_string(vars(args), f"{mode}_gt_file", assigned_keys=['dataset', 'lang'])
        evaluator = CaptionEvaluator(
            loader=loader,
            gt_file_path=gt_file_path,
            evaluation_settings=evaluation_settings,
            mode=mode,
            logger=logger
        )

        evaluator(model, output_path=output_path, no_score=args.no_score, print_sent=args.print_sent)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Time {}'.format(total_time_str))

'''
python infer_caption.py --model zeronlg-4langs-vc --output_path output/zeronlg-4langs-vc --dataset msrvtt --lang en --modes val test
'''
