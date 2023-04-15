import os
import logging
import torch.nn as nn
import argparse

from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler
from sentence_transformers.models import Pooling

from zeronlg import Framework
from zeronlg.losses import LossManager
from zeronlg.datasets import PretrainDataset
from zeronlg.models import Dense, Projector, Decoder, Transformer

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model settings
    parser.add_argument('--teacher_model_name', type=str, default='clip-ViT-B-32', 
                        choices=['clip-ViT-B-32', 'clip-ViT-B-16', 'clip-ViT-L-14'], 
                        help='Monolingual teacher model')
    parser.add_argument('--student_model_name', type=str, default='distilbert-base-multilingual-cased',  
                        help='Multilingual student model we use to imitate the teacher model\' outputs')
    parser.add_argument('--decoder_name', type=str, default='bert-base-multilingual-cased', 
                        help='Multilingual student model for NLG')
    # Student's encoder settings
    parser.add_argument('--max_seq_length', type=int, default=128, 
                        help='Student model max. lengths for inputs (number of word pieces)')
    
    # Student's decoder settings
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--use_pretrained_decoder', action='store_true', 
                        help='Whether to load pre-trianed weights of the decoder; \
                              Note that we will add randomly initialized cross-attention layers to the decdoder, \
                              even if you specify `use_pretrained_decoder` to True')
    parser.add_argument('--decoder_max_seq_length', type=int, default=128, 
                        help='Student model max. lengths for decoder inputs (number of word pieces)')

    # Data settings
    parser.add_argument('--train_corpus_format', type=str, default="data/corpus/multilingual_cc3m/4langs/cc3m_{}-{}.tsv")
    parser.add_argument('--source_language', type=str, default='en', choices=['en'], 
                        help='Our teacher model accepts English (en) sentences')
    parser.add_argument('--target_languages', type=str, nargs='+', default=['en', 'zh', 'de', 'fr'], 
                        help='The languages to be learned by the student model')
    parser.add_argument('--max_sentences', type=int, help='maximun number of sentences per file')
    parser.add_argument('--weights', type=int, nargs='+', help='If more than one dataset is loaded with load_data: With which frequency should data be sampled from this dataset?')
    parser.add_argument('--numpy_path', type=str, help='Path to a numpy file that stores sentence embeddings')
    parser.add_argument('--num_workers', type=int, default=4, help='# workers to load data; only activated when `numpy_path` is specified')

    # Training settings
    parser.add_argument('--use_amp', action='store_true', help='Whether use automatic mixed precision (amp) to speed up training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--inference_batch_size', type=int, help='Batch size at inference; if not speficied, set to batch_size')
    parser.add_argument('--epochs', type=int, default=3, help='Train for x epochs')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Warumup steps')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='warmupconstant', 
                        choices=['constantlr', 'warmupconstant', 'warmuplinear', 'warmupcosine', 'warmupcosinewithhardrestarts'])

    # Output settings
    parser.add_argument('--output_path', type=str, help='The exact output path to save training info and checkpoints')
    parser.add_argument('--output_root', type=str, default='output/2stages')
    parser.add_argument('--exp_name', type=str, default='debug', help='Experiment name; If `output_path` is not specified, the output path will be {output_root}/{exp_name}')
    parser.add_argument('--log_every', type=int, default=500)
    
    # Method-related settings
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0, 0.0, 0.0, 0.0], 
                        help='Scales of loss_mse, loss_at_teacher, loss_at_student, loss_contrastive')
    parser.add_argument('--use_masking', action='store_true',
                        help='Wheter to apply input corruption, i.e., randomly mask encoder\'s input tokens')
    parser.add_argument('--mask_prob', type=float, default=0.15,
                        help='Probability to mask tokens')
    parser.add_argument('--noise_std', type=float, default=0,
                        help='Standard deviation of gaussian noise; 0 means not applying feature corruption')
    parser.add_argument('--noise_prob', type=float, default=0, 
                        help='Probability to add gaussian noise; only activated when it is larger than 0')
    parser.add_argument('--student_emb_keyname', type=str, default='sentence_embedding', 
                        choices=['sentence_embedding', 'token_embeddings'],
                        help='If set to `sentence_embedding`, decoder will generate texts solely based on a global vector; \
                              Otherwise, decoder will base text generation on a sequence of token features. \
                              We find that set this arg to `token_embeddings` benefit machine translation a lot')
    
    # Model settings
    parser.add_argument('--no_frozen', action='store_true')
    parser.add_argument('--freeze_transformer', action='store_true')
    args = parser.parse_args()

    if not args.inference_batch_size:
        args.inference_batch_size = args.batch_size
    
    assert len(args.scales) == 4

    output_path = args.output_path or os.path.join(args.output_root, args.exp_name)
    os.makedirs(output_path, exist_ok=True)

    # saving training logs to {output_path}/log.txt
    logger.addHandler(logging.FileHandler(os.path.join(output_path, 'log.txt'), 'w', encoding='utf-8'))

    # log necessary information
    logger.info(f"Output path: {output_path}")
    logger.info(f"Target languages: {args.target_languages}")
    logger.info(f'Loss scales: {args.scales}')
    logger.info(f'Noise std: {args.noise_std} {f"(noise prob: {args.noise_prob})" if args.noise_prob > 0 else ""}')
    if args.use_amp:
        logger.info('Use amp for speeding up training')
    if args.use_masking:
        logger.info(f'Random masking: {args.use_masking} (prob: {args.mask_prob})')

    ######## Teacher model ########
    logger.info(f"Load teacher model: {args.teacher_model_name}")
    teacher_model = Framework(args.teacher_model_name)
    logger.info(f'Teacher model architecture: \n {teacher_model}')

    # freeze teacher model
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    dim_teacher = teacher_model._last_module().get_sentence_embedding_dimension()

    ######## Student model ########
    logger.info(f"Create student model from {args.student_model_name}")

    if args.student_model_name in ['distilbert-base-multilingual-cased']:
        # a transformer model for encoding
        encoder = Transformer(
            args.student_model_name, 
            max_seq_length=args.max_seq_length,
        )
        dim_enc = encoder.get_word_embedding_dimension()

        pooling_model = Pooling(dim_enc)
        dense_model = Dense(dim_enc, dim_teacher, bias=False, activation_function=nn.modules.linear.Identity())
        modules = [encoder, pooling_model, dense_model]
    else:
        student_model = Framework(args.student_model_name, load_sbert_only=True)
        modules = student_model.get_modules()

    attend_to = []
    if args.scales[1]:
        attend_to.append('teacher')
    if args.scales[2]:
        attend_to.append('student')
    
    if isinstance(modules[-1], Dense):
        # only encoding modules included now
        dim_student = modules[-1].get_sentence_embedding_dimension()
        assert dim_teacher == dim_student
        assert isinstance(modules[-1], Dense)

        # check if we need to add decoding modules
        if args.scales[1] or args.scales[2]:

            if 'bert' not in args.decoder_name.lower():
                raise NotImplementedError('You should take care of `num_hidden_layers`')

            decoder = Decoder(
                model_name_or_path=args.decoder_name,
                model_args={
                    'is_decoder': True, 
                    'add_cross_attention': True,
                    'num_hidden_layers': args.num_hidden_layers},
                from_pretrained=args.use_pretrained_decoder,
                attend_to=attend_to,
                teacher_model_name=args.teacher_model_name,
                max_seq_length=args.decoder_max_seq_length,
            )
            dim_dec = decoder.get_word_embedding_dimension()

            projector = Projector(dim_teacher, dim_dec, noise_std=args.noise_std, noise_prob=args.noise_prob, student_emb_keyname=args.student_emb_keyname)
            modules.extend([projector, decoder])
    else:
        # both encoding and decoding modules included
        assert student_model.get_module_attribute('teacher_model_name') == args.teacher_model_name

        student_model.set_module_attribute(Projector, 'noise_std', args.noise_std)
        
        # check if we need to keep decoding modules
        if args.scales[1] or args.scales[2]:
            student_model.set_module_attribute(Decoder, 'attend_to', attend_to)
        else:
            logger.info('Training does not need the decoder, ignore it')
            modules = student_model.get_encoding_modules()

    if args.scales[0] == 0 and not args.no_frozen:
        logger.info('Freeze the multimodal encoder of the student model')
        for idx, module in enumerate(modules):
            if isinstance(module, Projector):
                break
        for module in modules[:idx]:
            for p in module.parameters():
                p.requires_grad = False
    elif args.scales[0] == 0 and args.freeze_transformer:
        logger.info('Freeze the transformer of the multimodal encoder of the student model')
        module = modules[0]
        assert isinstance(module, Transformer)
        for p in module.parameters():
            p.requires_grad = False

    student_model = Framework(modules=modules, logger=logger)
    student_model.set_module_attribute(Dense, 'proj_token_embs', args.student_emb_keyname == 'token_embeddings')

    if args.scales[0] == 0 and args.scales[1] == 0 and args.scales[3] == 0:
        logger.info('Training does not need the teacher model, set it to None')
        teacher_model = None
    
    if args.scales[0] == 0 and args.scales[2] == 0 and args.scales[3] == 0:
        logger.info('Training does not need the multimodal encoder, ignore it')
        student_model = Framework(modules=student_model.get_decoding_modules(), logger=logger)

    logger.info(f'Student model architecture: \n {student_model}')
    logger.info(f"Total Params: {sum(p.numel() for p in student_model.parameters())}")
    logger.info(f"Trainable Params: {sum(p.numel() for p in student_model.parameters() if p.requires_grad)}")

    ###### Read Parallel Sentences Dataset ######
    train_data = PretrainDataset( 
        teacher_model=teacher_model, 
        batch_size=args.inference_batch_size, 
        use_embedding_cache=True,
        logger=logger,
        numpy_path=args.numpy_path,
    )

    if not args.weights:
        args.weights = [100] * len(args.target_languages)
    else:
        assert isinstance(args.weights, list)
        if len(args.weights) == 1:
            args.weights = args.weights * len(args.target_languages)
        else:
            assert len(args.weights) == len(args.target_languages)

    for lang, weight in zip(args.target_languages, args.weights):
        train_corpus = args.train_corpus_format.format(args.source_language, lang)
        if lang == args.source_language:
            langs=[lang]
            train_corpus = train_corpus.replace(f'-{args.source_language}.', '.')
            train_data.load_data(train_corpus, max_sentences=args.max_sentences, max_sentence_length=None, exclude_source=False, langs=langs, weight=weight)
        else:
            langs = [args.source_language, lang]
            # we set exclude_source to True, because we do not want too many samples in the source language for training
            train_data.load_data(train_corpus, max_sentences=args.max_sentences, max_sentence_length=None, exclude_source=True, langs=langs, weight=weight)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers if args.numpy_path else 0)

    ###### Define the training objective ######
    train_loss = LossManager(student_model, *args.scales)

    ###### Start training ######
    student_model.fit(train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_params= {'lr': args.lr, 'eps': args.eps},
        weight_decay=args.weight_decay,
        output_path=output_path,
        checkpoint_path=output_path,
        checkpoint_save_steps=None, # save checkpoints every epoch, rather than spcific number of steps
        log_every=args.log_every,
        use_amp=args.use_amp,
        scheduler=args.scheduler,
        seed=args.seed,
        use_masking=args.use_masking,
        mask_prob=args.mask_prob,
    )
