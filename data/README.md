# ZeroNLG

## TOC

- [ZeroNLG](#zeronlg)
  - [TOC](#toc)
  - [Data Structure](#data-structure)
  - [Langauges and Datasets](#langauges-and-datasets)
  - [Data Formats](#data-formats)
    - [**Multilingual corpora for text-only training**](#multilingual-corpora-for-text-only-training)
    - [**Visual Captioning**](#visual-captioning)
    - [**Machine Translation**](#machine-translation)
    - [**Retrieval**](#retrieval)
  - [Notes for the cache folder](#notes-for-the-cache-folder)
  - [Notes for captioning](#notes-for-captioning)



## Data Structure
```
ZeroNLG/
    data
    ├── corpus                      # for text-only training
    │  └── multilingual_cc3m        # obtained by CC3M + Google Translator
    │       └── 4langs   
    │           ├── cc3m_en.tsv         # 1.1M English sentences
    │           ├── cc3m_en-zh.tsv      # 1.1M English-Chinese pairs
    │           ├── cc3m_en-de.tsv      # 1.1M English-German pairs
    │           └── cc3m_en-fr.tsv      # 1.1M English-French pairs
    ├── annotations                 # for evaluations or supervised training (finetuning)
    │   ├── $VisualCaptionDataset   # e.g., if this dataset has both English and Chinese annotations
    │   │   ├── en                  # English annotations
    │   │   │   ├── subsets         # for semi-supervised training
    │   │   │   │    ├─ 0.1%_0.json # a 0.1% subset of train.json 
    │   │   │   │    ├─ 0.1%_1.json # different seed
    │   │   │   │    ├─ 0.1%_2.json # totally 3 seeds
    │   │   │   │    ├─ ...
    │   │   │   │    └─ 10%_2.json  # a 10% subset of train.json 
    │   │   │   ├── train.json          
    │   │   │   ├── val.json
    │   │   │   ├── val_gt.json
    │   │   │   ├── test.json     
    │   │   │   └── test_gt.json     
    │   │   └── zh                  # Chinese annotations
    │   │       ├── subsets         # for semi-supervised training
    │   │       │    └─ ...  
    │   │       ├── train.json 
    │   │       ├── val.json
    │   │       ├── val_gt.json
    │   │       ├── test.json     
    │   │       └── test_gt.json  
    │   └── $TranslateDataset
    │       ├── en-zh               # English-Chinese Translation
    │       │   ├── train.en        
    │       │   ├── train.zh        
    │       │   ├── val.en          
    │       │   ├── val.zh          
    │       │   ├── test.en         ## required for zero-shot transfer
    │       │   └── test.zh         ## required for zero-shot transfer
    │       └── ...      
    ├── folders that store raw images of visual caption datasets (see configs.image_video_root)
    └── folders that store raw videos of visual caption datasets (see configs.image_video_root)
```
- You can download `corpus.zip` (165.6 MB) from [Google Drive](https://drive.google.com/file/d/1yCLpDLDO5TnoqfyHKwgi51Fw66QliOvM/view?usp=share_link) or [Baidu网盘](https://pan.baidu.com/s/1wQ6-3QJqFugVfK2lQ5oFAg)(extract code: `wx4a`)
- You can download `annotations.zip` (121.4 MB) from [Google Drive](https://drive.google.com/file/d/19n62ho2uPkJl-wPQoOvCzQdKy7AjvqaT/view?usp=share_link) or [Baidu网盘](https://pan.baidu.com/s/1SjujMs7Jo1dQu2Y1csezzg)(extract code: `4trx`).


**Notes:** See `prepare_text_data.ipynb` to know how we prepare annotations.

## Langauges and Datasets
Languages and their [ISO 639‑1 Code](https://www.iso.org/iso-639-language-codes.html):
- English (`en`)
- Chinese (`zh`)
- German (`de`)
- French (`fr`)
- Czech (`cs`) -- not used at all
- Japanese (`ja`) -- not used at all

We provide the following annotations:
- MS-COCO: `en`, `ja`
- Flickr30k: `en`, `zh`, `de`, `fr`, `cs`
- MSR-VTT: `en`
- VATEX: `en`, `zh`

**Notes:** See `prepare_text_data.ipynb` for more details, where we provide bibtex of references and etc. We can not re-distribute the raw images and videos, you should download them by yourself.


## Data Formats
### **Multilingual corpora for text-only training**
Filenames follow the format `{prefix}{sourceLang}-{targetLang}.tsv` if the source language is different from the target language. Taking `cc3m_en-zh.tsv` as an example (here, `prefix`="cc3m_", `sourceLang`="en", `targetLang`="zh"), the file content looks like:
```
english sentence 1      chinese sentence 1
english sentence 2      chinese sentence 2
...
```
where there are two columns separated by `\t` (Tab) and one line is for each sentence pair. 

Filenames follow the format `{prefix}{sourceLang}.tsv` if the source language equals to the target language. Taking `cc3m_en.tsv` as an example, the file content looks like:
```
english sentence 1
english sentence 2
...
```
where there is only one column and one line is for each sentence.

### **Visual Captioning**
Each dataset has 5 files: 
- `train.json`:
```
# a list of samples, each of which is a dict
[ 
  # a sample for video captioning
  # if a video is paired with N captions, then there will be N lines (samples)
  { "image": "all_videos/video0.mp4", # path relative to the video root, see configs.image_video_root
    "caption": "a car is shown",      # caption (in any language)
    "image_id": 0,                    
  }, 

  # a sample for image captioning
  { "image": "val2014/COCO_val2014_000000522418.jpg",
    "caption": "A woman wearing a net on her head cutting a cake.",
    "image_id": 522418,              
  }, 
  ...
] 
```
- `val.json`, 
```
# a list of samples, each of which is a dict
[ 
  # a sample for video captioning
  { "image": "all_videos/video6513.mp4", 
    "caption": [
      "a family is having coversation", 
      "a girl sings i wish that i could be like the cool kids", 
      "a music video about a social situation", 
      "a music video featuring a girl and a sunset for the song cool kids", 
      "a music video where a nerdy girl wishes she was like the cool kids",
      ...
    ] # includes all captions paired with this video
    "image_id": 6513
  }

  # a sample for image captioning
  { "image": "val2014/COCO_val2014_000000184613.jpg",
    "caption": [
      "A child holding a flowered umbrella and petting a yak.", 
      "A young man holding an umbrella next to a herd of cattle.", 
      "a young boy barefoot holding an umbrella touching the horn of a cow", 
      "A young boy with an umbrella who is touching the horn of a cow.", 
      "A boy holding an umbrella while standing next to livestock."
    ], # includes all captions paired with this image
    "image_id": 184613
  }, 
  ...
] 
```
- `val_gt.json`: We should prepare a ground-truth file accepted by COCO Evaluation Server like this:
```
# data/annotations/flickr30k/en/val_gt.json
{
  "annotations": [
    {
      "image_id": 1018148011,
      "caption": "A group of people stand in the back of a truck filled with cotton.",
      "id": 0,   # this corresponds to unique sentence id, whose value will not influence the evaluation.
    },
    {
      "image_id": 1018148011,
      "caption": "Men are standing on and about a truck carrying a white substance.",
      "id": 1,
    },
    ...,
    {
      'image_id': 1029450589, 
      'caption': 'An adult wearing a gray shirt with red sleeves sleeping on a couch.', 
      'id': XXXX,
    },
    ...
  ],
  "images": [
    {'id': 1018148011}, # image_id
    {'id': 1029450589},
    ...
  ] # includes all unique image id appearing at "annotations"
}
```
- `test.json`: similar to `val.json`
- `test_gt.json`. similar to `val_gt.json`

Note that for non-English annotations, we name the above 5 files with the `{lang}_` prefix, where `lang` belongs to {`zh`, `de`, `fr`} in the paper.

### **Machine Translation**
Each translation task of a specific dataset contains files like `{split}.{language1}` and `{split}.{language2}`, where `split` belongs to {`train` (optional), `val` (optional), `test`}. Within files, one line is for each sentence. More importantly, sentences on the same line in two files of the same split are paired data.

### **Retrieval**
Image/video-text retrieval needs three files `train.json`, `val.json`, and `test.json`, which are identical to the ones required in visual captioning.

## Notes for the cache folder
please set `ZERONLG_HOME` in the environment (e.g., `export ZERONLG_HOME=data/checkpoints`) to specify the cache folder that stores pre-trained models. If not specified, the default cache folder will be `~/.cache/torch/zeronlg`.

## Notes for captioning
1. To evaluate visual captioning, ensure that your system has installed JAVA. Here is an example:
   - Download from the official website (https://www.java.com/en/download/manual.jsp) to obatin, e.g., jdk-8u333-linux-x64.tar.gz
   - Unzip the file by running `tar -zxvf jdk-8u333-linux-x64.tar.gz`, and you will see the jre folder
   - Write the follwing lines to ~/.bashrc:
     - `echo "export JRE_HOME=path/to/jre" >> ~/.bashrc`
     - `echo "export PATH=${JRE_HOME}/bin:$PATH" >> ~/.bashrc`
   - Activate the settings by running `source ~/.bashrc`
   - See if the java has been installed: `java -version`
2. You should install packages `pycocoevalcap` and `pycocotools` (included in `requirement.txt`).
3. When calculating the `SPICE` metric, the code will try to automatically download two files `stanford-corenlp-3.6.0.jar` and `stanford-corenlp-3.6.0-models.jar`, and save them to ${pycocoevalcapPath}/spice/lib/. If you encounter a network issue, you can prepare these two files by yourself:
   - Download a zip file from https://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip
   - Unzip it to get the above two files
   - Run `pip show pycocoevalcap` to see where the package has been installed
   - Move the two files to ${pycocoevalcapPath}/spice/lib/
4. To evaluate visual captioning in Chinese, you should install the `jieba` package (included in `requirement.txt`) to tokenize Chinese sentences.
5. To evaluate visual captioning in German or in French, you should install the `stanfordcorenlp` package (included in `requirement.txt`) and download [stanford-corenlp-4.5.2](https://stanfordnlp.github.io/CoreNLP/history.html). You can simply run `python tests/test_eval_caption.py`, which will download requested files automatically.
