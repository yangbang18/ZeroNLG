# ZeroNLG

PyTroch implementation of our paper published in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 2024:
> **ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation**
> 
> Bang Yang, Fenglin Liu, Yuexian Zou, Xian Wu, Yaowei Wang, and David A. Clifton.
>
> [[arXiv]](https://arxiv.org/abs/2303.06458) [[TPAMI]](https://ieeexplore.ieee.org/document/10453989)  


## TOC

- [ZeroNLG](#zeronlg)
  - [TOC](#toc)
  - [Update Notes](#update-notes)
  - [Environment](#environment)
  - [Quick Start](#quick-start)
  - [Zero-Shot Performance](#zero-shot-performance)
    - [Visual captioning](#visual-captioning)
    - [Machine translation](#machine-translation)
  - [Reproduction](#reproduction)
    - [Data](#data)
    - [Training](#training)
    - [Testing (Zero-Shot Transfer)](#testing-zero-shot-transfer)
    - [Semi-Supervised Training on Visual Captioning](#semi-supervised-training-on-visual-captioning)
    - [Visualization and More](#visualization-and-more)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

## Update Notes
**[2023-12-01]** Release notebooks and upgrade to `zeronlg==1.0.1`

**[2023-04-06]** Release the code, data, and pre-trained models

## Environment
```bash
# clone the repo
git clone https://github.com/yangbang18/ZeroNLG

# enter the repo
cd ZeroNLG

# install a proper version of PyTorch
# see https://pytorch.org/get-started/previous-versions/
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# optional
pip install transformers==4.12.5

# install this repo with a editable mode
pip install -e .
```

**Note:** 
- We can run the code under `torch==1.13.1`, `torch==1.10.1` and `torch==1.8.1`. Other versions may work. 
- If you want to re-produce the inference results of our released pre-trained models, `transformers==4.12.5` is required.

## Quick Start
**Visual Captioning:**
```python
from zeronlg import ZeroNLG

# Automatically download models pre-trained for visual captioning from Huggingface Hub
model = ZeroNLG('zeronlg-4langs-vc')

# `images` can be a remote image url, a local image/video file, etc
# `lang` should be one of English ('en'), Chinese ('zh'), German ('de'), and French ('fr')
url='./asserts/dogs.webp'
model.forward(images=url, lang='en', num_beams=3, task='caption')
# ["dogs playing in the snow"]

model.forward(images=url, lang='zh', num_beams=3, task='caption')
# ["狗 在 雪 地 里 玩 耍"]

# Althernatively, you can call the specific forward function
model.forward_caption(images=url, lang='en', num_beams=3)
```

**Machine Translation**
```python
from zeronlg import ZeroNLG

# Automatically download models pre-trained for machine translation from Huggingface Hub
model = ZeroNLG('zeronlg-4langs-mt')

# Translating English into Chinese
# Note: the multilingual encoder is langauge-agnostic, so the `lang` below means the langauge to be generated
model.forward_translate(texts='a girl and a boy are playing', lang='zh', num_beams=3)
# ["一 个 女 孩 和 一 个 男 孩 一 起 玩"]
```

## Zero-Shot Performance
### Visual captioning
Model: [zeronlg-4langs-vc](https://huggingface.co/yangbang18/zeronlg-4langs-vc)'s multilingual decoder + CLIP's ViT-B-32 image encoder.
| Dataset | Language | Type | BLEU@1 | BLEU@2 | BLEU@3 | BLEU@4 | METEOR | ROUGE-L | CIDEr-D | SPICE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [Flickr30K](https://paperswithcode.com/paper/from-image-descriptions-to-visual-denotations) | English | Image | 46.4 | 27.2 | 15.5 | 8.9 | 13.0 | 31.3 | 21.0 | 7.6
| Flickr30K | [Chinese](https://dl.acm.org/doi/abs/10.1145/3123266.3123366) | Image | 45.3 | 25.5 | 14.6 | 8.4 | - | 31.8 | 18.0 | -
| Flickr30K | [German](https://github.com/multi30k/dataset) | Image | 41.9 | 21.1 | 11.2 | 5.7 | - | 21.2 | 17.1 | -
| Flickr30K | [French](https://github.com/multi30k/dataset) | Image | 19.8 | 9.5 | 5.0 | 2.8 | - | 18.6 | 24.8 | -
| [COCO](https://paperswithcode.com/paper/microsoft-coco-captions-data-collection-and) | English | Image | 47.5 | 29.0 | 16.8 | 9.6 | 14.4 | 34.9 | 29.9 | 8.7
| [MSR-VTT](https://paperswithcode.com/paper/msr-vtt-a-large-video-description-dataset-for) | English | Video | 52.2 | 31.9 | 16.6 | 8.7 | 15.0 | 35.4 | 9.9 | -
| [VATEX](https://paperswithcode.com/paper/vatex-a-large-scale-high-quality-multilingual) | English | Video | 42.2 | 24.6 | 12.5 | 6.3 | 11.7 | 29.3 | 9.1 | -
| VATEX | Chinese | Video | 41.9 | 24.3 | 13.7 | 7.1 | - | 29.6 | 9.8 | -

**Notes:**
- For non-English visual captioning, we do not report METEOR and SPICE, beacause they consider synonym matching and named entity recognition in English by default.
- For video captioning in English, we do not report SPICE following common practices.

### Machine translation
Model: [zeronlg-4langs-mt](https://huggingface.co/yangbang18/zeronlg-4langs-mt) only.

| Toolkit | En->Zh | En<-Zh | En->De | En<-De | En->Fr | En<-Fr | Zh->De | Zh<-De | Zh->Fr | Zh<-Fr | De->Fr | De<-Fr|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
**SacreBLEU**|14.7|8.8|20.5|21.1|22.0|24.6|7.3|11.9|5.2|16.2|16.7|18.5
**NLKT**|6.0|9.2|21.6|23.2|27.2|26.8|7.8|4.6|6.1|9.7|20.9|19.6

**Notes**:
- `Metric`: BLEU.
- Following the common practice in machine translation, we use `SacreBLEU` rather than `NLTK` to measure BLEU.
- `A->B` means translating A language into B language.
- `A<-B` means translating B language into A language.


## Reproduction
### Data
Please see [data/README.md](/data/) for more details.

### Training
The training process does not involve any validation operation, i.e., you should choose the best-performed model on a specific dataset by yourself. **In our experiments, we always use ZeroNLG that applys auto-encoding training after 3 epochs for downstream evaluations because it generally performs the best.**

**Stage 1**: Cross-Lingual Alignment

```shell
# require 2 hours with a 24-GB GTX4090 
# (3 epochs, batch_size 128)
python train.py \
--use_amp \
--scales 1 0 0 0 \
--target_languages en zh de fr \
--student_model_name distilbert-base-multilingual-cased \
--output_path output/1_mDistilBERT \
--batch_size 128
```
Here, we showcase how to use pre-trained multilingual DistilBERT as an starting point to train the multilingual encoder. There are all options for `student_model_name`:
- `distilbert-base-multilingual-cased` (vocab size: 119547)
- `bert-base-multilingual-cased` (vocab size: 119547)
- `xlm-roberta-base` (vocab size: 250002)
- `sentence-transformers/clip-ViT-B-32-multilingual-v1` (vocab size: 119547)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (vocab size: 250002) 

**Stage 2**: Denoising Language Reconstruction (Visual Captioning)

```shell
# require 6.5 hours with a 24-GB GTX4090 
# (3 epochs, batch_size 32)
python train.py \
--use_amp \
--scales 0 0 1 0 \
--target_languages en zh de fr \
--student_model_name output/1_mDistilBERT \
--output_path output/2_ZeroNLG_VC \
--no_tie_all \
--init_word_embeddings \
--noise_std 0.1
```

**Note:** 
- Use `python train.py --help` to see explainations of arguments.
- In this case, decoder's embeddings are initialized as (but not tied to) that of the encoder. This produces better performance than tying encoder's and decoder's embeddings, though introducing more parameters.
- When the multilingual decoder does not use a mBERT-style tokenizer (e.g., using XLM-RoBERTa-style), please specify the argument `--language_identifier_strategy` with `type` rather than the default choise (`bos`).

**Stage 2**: Denoising Language Reconstruction (Machine Translation)

```shell
# require 6.5 hours with a 24-GB GTX4090 
# (3 epochs, batch_size 32)
python train.py \
--use_amp \
--scales 0 0 1 0 \
--target_languages en zh de fr \
--student_model_name output/1_mDistilBERT \
--output_path output/2_ZeroNLG_MT \
--student_emb_keyname token_embeddings \
--use_masking \
--mask_prob 0.05 \
--noise_std 0.001
```

**Note:** 
- Use `python train.py --help` to see explainations of arguments.
- We found that auto-encoding on token-level (rather than sentence-level) embeddings produced better performance for machine translation.

### Testing (Zero-Shot Transfer)
```bash
# visual captioning
## evluate the model trained after 3 epochs
## `output/2_ZeroNLG_VC` is equivalent to `output/2_ZeroNLG_VC/2`
export model=output/2_ZeroNLG_VC
bash scripts/caption.sh $model

## evluate the model trained after 1 epoch
export model=output/2_ZeroNLG_VC/0
bash scripts/caption.sh $model

# machine translation
export model=output/2_ZeroNLG_MT
bash scripts/translate.sh $model

# retrieval
export model=output/2_ZeroNLG_VC
bash scripts/retrieval.sh $model
```

### Semi-Supervised Training on Visual Captioning
```bash
# training on limited labeled data w/o pre-training
bash scripts/semi.sh coco en
bash scripts/semi.sh msrvtt en
bash scripts/semi.sh flickr30k de
bash scripts/semi.sh flickr30k fr
bash scripts/semi.sh vatex zh

# training on limited labeled data w/ pre-training
export model=output/2_ZeroNLG_VC
bash scripts/semi.sh coco en $model
bash scripts/semi.sh msrvtt en $model
bash scripts/semi.sh flickr30k de $model
bash scripts/semi.sh flickr30k fr $model
bash scripts/semi.sh vatex zh $model
```
The script will loop over `0.01%` (if available), `0.1%`, `1%`, and `10%` labeled data, each for three times (as we generate subsets with 3 different seeds).

### Visualization and More
Please refer to [notebooks](/notebooks/).

## Bugs or Questions?

If you encounter any problems when using the code, or want to report a bug, you can open an issue or email yangbang@pku.edu.cn, fenglin.liu@eng.ox.ac.uk. Please try to specify the problem with details so we can help you better and quicker!


## Citation

Please consider citing our papers if our code, data and models are useful to your work, thanks sincerely!

```bibtex
@misc{Yang2023ZeroNLG,
   title={ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation},
   author={Yang, Bang and Liu, Fenglin and Zou, Yuexian and Wu, Xian and Wang, Yaowei and Clifton, David A.},
   journal={arXiv preprint arXiv:2303.06458}
   year={2023},
   eprint = {2303.06458},
   archiveprefix = {arxiv}
}
```

## Acknowledgements
Our code is built upon [sentence-transformers](https://github.com/UKPLab/sentence-transformers).
