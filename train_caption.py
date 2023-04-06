
import os
import argparse
import logging
import torch
import numpy as np
import configs

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler

import zeronlg
from zeronlg import CaptionDataset, CaptionEvaluator
from zeronlg.models import Projector, Decoder, CLIPModel
from zeronlg.utils import get_formatted_string
from zeronlg.losses import LossManager

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

try:
    ROOT = configs.annotation_caption_root
except:
    ROOT = configs.annotation_root


class Framework(zeronlg.Framework):
    def smart_batching_collate(self, batch):
        texts = [b['text'] for b in batch]
        langs = [b['lang'] for b in batch]
        embs = np.array([b['emb'] for b in batch])
        
        features = {}
        features.update(self.decoder_tokenize(texts, langs))
        features['source_embedding'] = torch.FloatTensor(embs)

        labels = None
        return features, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--teacher_model_name', type=str, default='clip-ViT-B-32', 
                    choices=['clip-ViT-B-32', 'clip-ViT-B-16', 'clip-ViT-L-14'], 
                    help='Monolingual teacher model')
    parser.add_argument('--use_clip_tokens', type=int, help='Whether use token-level visual embeddings?')
    parser.add_argument('--mean_pooling', action='store_true', help='average visual embeddings over the time axis')

    parser.add_argument('--decoder_name', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--use_pretrained_decoder', action='store_true')
    parser.add_argument('--decoder_max_seq_length', type=int, default=128, help='Student model max. lengths for inputs (number of word pieces)')
    parser.add_argument('--freeze_word_embs', action='store_true')

    # Data paths and attributes
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--lang', type=str, default='en', help='Language')
    parser.add_argument('--data_root', type=str, default=ROOT)
    parser.add_argument('--train_file', type=str, help='If not specified, use train_file_format')
    parser.add_argument('--val_file', type=str, help='If not specified, use val_file_format')
    parser.add_argument('--val_gt_file', type=str, help='If not specified, use val_gt_file_format')
    parser.add_argument('--test_file', type=str, help='If not specified, use test_file_format')
    parser.add_argument('--test_gt_file', type=str, help='If not specified, use test_gt_file_format')
    parser.add_argument('--pickle_path', type=str, help='If not specified, use pickle_path_format')
    parser.add_argument('--subset', type=str)
    parser.add_argument('--train_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/train.json'))
    parser.add_argument('--val_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/val.json'))
    parser.add_argument('--val_gt_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/val_gt.json'))
    parser.add_argument('--test_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/test.json'))
    parser.add_argument('--test_gt_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/test_gt.json'))
    parser.add_argument('--pickle_path_format', type=str, default=os.path.join(ROOT, '{dataset}/{clip_model_name}_{mode}{postfix}.pkl'))
    parser.add_argument('--subset_path_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/subsets/{subset}.json'))

    # Training settings
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Warumup steps')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='warmupconstant', choices=['constantlr', 'warmupconstant', 'warmuplinear', 'warmupcosine', 'warmupcosinewithhardrestarts'])
    parser.add_argument('--auto', action='store_true')
    
    # Evaluation settings
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=30)
    parser.add_argument('--min_length', type=int, default=5)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    # Output settings
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--log_every', type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(args.output_path, 'log.txt'), 'w', encoding='utf-8'))

    if args.subset:
        args.train_file_format = args.subset_path_format

    ##############################################################################
    logger.info('Creating models')
    if args.model:
        logger.info(f'Load the student model from {args.model}')
        student_model = Framework(args.model)
        student_model.set_module_attribute(Projector, 'noise_std', 0.0)
        student_model.set_module_attribute(Decoder, 'attend_to', ['teacher'])

        use_clip_tokens = bool(args.use_clip_tokens or student_model.get_module_attribute('use_clip_tokens', False))
        student_model.set_module_attribute(Decoder, 'use_clip_tokens', use_clip_tokens)

        student_model = Framework(
            modules=student_model.get_decoding_modules(), 
            freeze_word_embeddings=args.freeze_word_embs,
            logger=logger
        )
        teacher_model_name = student_model.get_module_attribute('teacher_model_name')
        logger.info(f'Load the teacher model from {teacher_model_name}')
        teacher_model = Framework(teacher_model_name)
        teacher_model.set_module_attribute(CLIPModel, 'use_clip_tokens', use_clip_tokens)
    else:
        teacher_model_name = args.teacher_model_name
        logger.info(f'Load the teacher model from {teacher_model_name}')
        teacher_model = Framework(teacher_model_name)
        teacher_model.set_module_attribute(CLIPModel, 'use_clip_tokens', args.use_clip_tokens)
        
        logger.info(f'Create the randomly initialized student model')
        decoder = Decoder(
            model_name_or_path=args.decoder_name,
            model_args={
                'is_decoder': True, 
                'add_cross_attention': True,
                'num_hidden_layers': args.num_hidden_layers,
                'hidden_dropout_prob': args.hidden_dropout_prob},
            from_pretrained=args.use_pretrained_decoder,
            attend_to=['teacher'],
            teacher_model_name=teacher_model_name,
            use_clip_tokens=args.use_clip_tokens,
            max_seq_length=args.decoder_max_seq_length,
        )
        dim_tea = teacher_model._last_module().get_sentence_embedding_dimension()
        dim_dec = decoder.get_word_embedding_dimension()

        projector = Projector(dim_tea, dim_dec)
        student_model = Framework(modules=[projector, decoder], logger=logger)

    logger.info(f'Student model architecture: \n {student_model}')
    logger.info(f"Total Params: {sum(p.numel() for p in student_model.parameters())}")
    logger.info(f"Trainable Params: {sum(p.numel() for p in student_model.parameters() if p.requires_grad)}")

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    ##############################################################################
    logger.info('Creating dataloaders')

    loaders = []
    for mode in ['train', 'val', 'test']:
        ann_rpath = get_formatted_string(vars(args), f"{mode}_file", assigned_keys=['dataset', 'lang', 'subset'])

        pickle_path = get_formatted_string(vars(args), 'pickle_path', assigned_kwargs=dict(
            dataset=args.dataset, clip_model_name=teacher_model_name, mode=mode, postfix='_tokens' if student_model.get_module_attribute('use_clip_tokens', False) else ''
        ))

        dataset = CaptionDataset(
            vision_root=configs.image_video_root[args.dataset],
            ann_rpath=ann_rpath,
            lang=args.lang,
            clip_model=teacher_model,
            pickle_path=pickle_path,
            logger=logger,
            mean_pooling=args.mean_pooling
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True if mode == 'train' else False,
            collate_fn=dataset.collate_fn,
            # NOTE: do not set `num_workers`, it will raise an error about `using cuda with multiprocessing`
        )

        logger.info(f'{mode} #data: {len(dataset)}, #batches: {len(loader)}')
        loaders.append(loader)
    
    train_loader, val_loader, test_loader = loaders

    ##############################################################################
    evaluation_settings = {
        k: getattr(args, k) 
        for k in ['lang', 'num_beams', 'max_length', 'min_length', 'repetition_penalty']
    }
    if args.auto:
        evaluation_settings.update(configs.auto_settings[args.lang])

    logger.info(f'Evaluation settings: {evaluation_settings}')

    gt_file_path = get_formatted_string(vars(args), "val_gt_file", assigned_keys=['dataset', 'lang'])
    evaluator = CaptionEvaluator(
        loader=val_loader,
        gt_file_path=gt_file_path,
        evaluation_settings=evaluation_settings,
        mode='val',
        logger=logger,
        with_epoch=True,
    )

    train_loss = LossManager(student_model, loss_mse_scale=0, loss_at_teacher_scale=1)

    student_model.fit(train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        evaluator=evaluator,
        save_best_model=True,
        warmup_steps=args.warmup_steps,
        optimizer_params= {'lr': args.lr, 'eps': args.eps},
        weight_decay=args.weight_decay,
        output_path=args.output_path,
        log_every=args.log_every,
        use_amp=args.use_amp,
        scheduler=args.scheduler,
        seed=args.seed,
    )

    ##############################################################################
    gt_file_path = get_formatted_string(vars(args), "test_gt_file", assigned_keys=['dataset', 'lang'])
    evaluator = CaptionEvaluator(
        loader=test_loader,
        gt_file_path=gt_file_path,
        evaluation_settings=evaluation_settings,
        mode='test',
        logger=logger,
    )

    student_model = Framework(args.output_path)
    evaluator(student_model, os.path.join(args.output_path, 'eval'))

    train_loader.dataset.save_pickle()
    val_loader.dataset.save_pickle()
    test_loader.dataset.save_pickle()
