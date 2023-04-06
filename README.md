# ZeroNLG

PyTroch implementation of our Preprint paper:
> **ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation**
> 
> Bang Yang, Fenglin Liu, Yuexian Zou, Xian Wu, Yaowei Wang, and David A. Clifton.
>
> [[arXiv]](https://arxiv.org/abs/2303.06458)


## TOC

- [ZeroNLG](#zeronlg)
  - [TOC](#toc)
  - [Update Notes](#update-notes)
  - [Environment](#environment)
  - [Quick Start](#quick-start)
  - [Zero-Shot Performance](#zero-shot-performance)
    - [Visual captioning](#visual-captioning)
    - [Cross-modal retrieval](#cross-modal-retrieval)
    - [Machine translation](#machine-translation)
  - [Reproduction](#reproduction)
    - [Data](#data)
    - [Training](#training)
    - [Testing (Zero-Shot Transfer)](#testing-zero-shot-transfer)
    - [Semi-Supervised Training on Visual Captioning](#semi-supervised-training-on-visual-captioning)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

## Update Notes
[2023-04-06] We release the code, data, and pre-trained models.

## Environment
```bash
# clone the repo
git clone https://github.com/yangbang18/ZeroNLG

# enter the repo
cd ZeroNLG

# install a proper version of PyTorch
# see https://pytorch.org/get-started/previous-versions/
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# install this repo with a editable mode
pip install -e .
```

We can run the code under `torch==1.13.1`, `torch==1.10.1` and `torch==1.8.1`. Other versions may work.

## Quick Start
**Visual Captioning:**
```python
from zeronlg import ZeroNLG

# Automatically download the model from Huggingface Hub
# Note: this model is especially pre-trained for visual captioning
model = ZeroNLG('zeronlg-4langs-vc')

# `images` can be a remote image url, a local image/video file, etc
# `lang` should be one of English ('en'), Chinese ('zh'), German ('de'), and French ('fr')
url = 'https://img2.baidu.com/it/u=1856500011,1563285204&fm=253&fmt=auto&app=138&f=JPEG?w=667&h=500'
caption = model.forward(images=url, lang='en', num_beams=3, task='caption') 
# caption = "dogs play in the snow"

caption = model.forward(images=url, lang='zh', num_beams=3, task='caption') 
# caption = "狗 在 雪 地 里 玩 耍"

# Althernatively, you can call the specific forward function
caption = model.forward_caption(images=url, lang='en', num_beams=3)
```

**Machine Translation**
```python
from zeronlg import ZeroNLG

# Automatically download the model from Huggingface Hub
# Note: this model is especially pre-trained for machine translation
model = ZeroNLG('zeronlg-4langs-mt')

# Translating English into Chinese
# Note: the multilingual encoder is langauge-agnostic, so the `lang` below means the langauge to be generated
output = model.forward_translate(texts='a girl and a boy are playing', lang='zh', num_beams=3)
# output = "一 个 女 孩 和 一 个 男 孩 一 起 玩"
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

### Cross-modal retrieval
Model: [zeronlg-4langs-vc](https://huggingface.co/yangbang18/zeronlg-4langs-vc)'s multilingual encoder + CLIP's ViT-B-32 image encoder
| Dataset | Language | Type | I2T R@1 | I2T R@5 | I2T R@10 | I2T Mean | T2I R@1 | T2I R@5 | T2I R@10 | T2I Mean | Avg.|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [Flickr30K](https://paperswithcode.com/paper/from-image-descriptions-to-visual-denotations) | English | Image | 75.2 | 93.9 | 97.1 | 88.7 | 57.1 | 82.2 | 89.1 | 76.1 | 82.4|
| Flickr30K | [Chinese](https://dl.acm.org/doi/abs/10.1145/3123266.3123366) | Image | 75.0 | 93.0 | 96.7 | 88.2 | 53.8 | 79.8 | 87.1 | 73.6 | 80.9|
| Flickr30K | [German](https://github.com/multi30k/dataset) | Image | 70.9 | 91.1 | 95.7 | 85.9 | 47.5 | 74.1 | 83.1 | 68.2 | 77.1|
| Flickr30K | [French](https://github.com/multi30k/dataset) | Image | 55.8 | 83.4 | 91.5 | 76.9 | 56.6 | 81.2 | 88.4 | 75.4 | 76.2|
| [COCO 5K](https://paperswithcode.com/paper/microsoft-coco-captions-data-collection-and) | English | Image | 45.0 | 71.1 | 80.3 | 65.5 | 28.2 | 53.3 | 64.5 | 48.7 | 57.1
| COCO 1K | English | Image | 66.0 | 89.1 | 94.6 | 83.2 | 47.5 | 77.5 | 87.9 | 71.0 | 77.1 |
| [MSR-VTT](https://paperswithcode.com/paper/msr-vtt-a-large-video-description-dataset-for) | English | Video | 32.0 | 55.5 | 65.8 | 51.1 | 17.9 | 36.4 | 45.5 | 33.3 | 42.2
| [VATEX](https://paperswithcode.com/paper/vatex-a-large-scale-high-quality-multilingual) | English | Video | 26.9 | 52.8 | 64.2 | 48.0 | 19.2 | 41.2 | 52.7 | 37.7 | 42.8
| VATEX | Chinese | Video | 40.6 | 70.9 | 82.7 | 64.7 | 28.8 | 58.0 | 70.1 | 52.3 | 58.5 |

**Notes:**
- `I2T`: image-to-text retrieval, image as the query, search similar texts
- `T2I`: text-to-image retrieval, text as the query, search similar images
- `R@K`: Recall rate at top-K candidates
- `Avg.`: Average of `R@{1,5,10}` on both directions
- Retrieval uses the same testing sets as those for visual captioning, except `COCO-1K`, which splits the original testing set into 5 folds and report performance averaged over 5 folds. 

### Machine translation
Model: [zeronlg-4langs-mt](https://huggingface.co/yangbang18/zeronlg-4langs-mt) only.

| En->Zh | En<-Zh | En->De | En<-De | En->Fr | En<-Fr | Zh->De | Zh<-De | Zh->Fr | Zh<-Fr | De->Fr | De<-Fr|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
6.0|9.2|21.6|23.2|27.2|26.8|7.8|4.6|6.1|9.7|20.9|19.6

**Notes**:
- `Metric`: BLEU.
- `A->B` means translating A language into B language.
- `B<-A` means translating B language into A language.


## Reproduction
### Data
Please see [data/README.md](https://openi.pcl.ac.cn/yangb02/AdaCLIP/src/branch/pami/DATA.md) for more details.

### Training
Pre-training for visual captioning
```bash
# Stage 1: cross-lingual alignment using MSE
#          initialize multilingual encoder with Multilingual-CLIP
#          roughly 5 hours with a 16-GB Tesla T4 (3 epochs, batch_size 128)
python train.py \
--use_amp \
--scales 1 0 0 \
--target_languages en zh de fr \
--student_model_name sentence-transformers/clip-ViT-B-32-multilingual-v1 \
--output_path output/2stages/1_MSE \
--batch_size 128

# Stage 2: auto-encoding with feature corruption
#          roughly 13 hours with a 16-GB Tesla T4 (3 epochs, batch_size 32)
python train.py \
--use_amp \
--scales 0 0 1 \
--target_languages en zh de fr \
--student_model_name output/2stages/1_MSE \
--output_path output/2stages/2_ZeroNLG_VC \
--noise_std 0.1
```

Pre-training for machine translation
```bash
# Stage 1: cross-lingual alignment using MSE
#          initialize multilingual encoder with DistilBERT
#          roughly 5 hours with a 16-GB Tesla T4 (3 epochs, batch_size 128)
python train.py \
--use_amp \
--scales 1 0 0 \
--target_languages en zh de fr \
--student_model_name distilbert-base-multilingual-cased \
--output_path output/2stages/1_MSE_MT \
--batch_size 128

# Stage 2: auto-encoding with input and feature corruptions
#          roughly 13 hours with a 16-GB Tesla T4 (3 epochs, batch_size 32)
python train.py \
--use_amp \
--scales 0 0 1 \
--target_languages en zh de fr \
--student_model_name output/2stages/1_MSE_MT \
--output_path output/2stages/2_ZeroNLG_MT \
--student_emb_keyname token_embeddings \
--use_masking \
--mask_prob 0.05 \
--noise_std 0.001
```
**Note:** The training process does not involve any validation operation, i.e., you should choose the best-performed model on a specific dataset by yourself. **In our experiments, we always use ZeroNLG that applys auto-encoding training after 3 epochs for downstream evaluations because it generally performs the best.**


### Testing (Zero-Shot Transfer)
```bash
# visual captioning
## evluate the model trained after 3 epochs
## `output/2stages/2_ZeroNLG_VC` is equivalent to `output/2stages/2_ZeroNLG_VC/2`
export model=output/2stages/2_ZeroNLG_VC
bash scripts/caption.sh $model

## evluate the model trained after 1 epoch
export model=output/2stages/2_ZeroNLG_VC/0
bash scripts/caption.sh $model

# machine translation
export model=output/2stages/2_ZeroNLG_MT
bash scripts/translate.sh $model

# retrieval
export model=output/2stages/2_ZeroNLG_VC
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
export model=output/2stages/2_ZeroNLG_VC
bash scripts/semi.sh coco en $model
bash scripts/semi.sh msrvtt en $model
bash scripts/semi.sh flickr30k de $model
bash scripts/semi.sh flickr30k fr $model
bash scripts/semi.sh vatex zh $model
```
The script will loop over `0.01%` (if available), `0.1%`, `1%`, and `10%` labeled data, each for three times (as we generate subsets with 3 different seeds).

## Bugs or Questions?

If you encounter any problems when using the code, or want to report a bug, you can open an issue or email yangbang@pku.edu.cn, fenglin.liu@eng.ox.ac.uk. Please try to specify the problem with details so we can help you better and quicker!


## Citation

Please consider citing our papers if our code, data and models are useful to your work, thanks sincerely!

```bibtex
@article{Yang2023ZeroNLG,
   title={ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation},
   author={Yang, Bang and Liu, Fenglin and Zou, Yuexian and Wu, Xian and Wang, Yaowei and Clifton, David A.},
   journal={arXiv preprint arXiv:2303.06458}
   year={2023}
}
```

## Acknowledgements
Our code is built upon [sentence-transformers](https://github.com/UKPLab/sentence-transformers).
