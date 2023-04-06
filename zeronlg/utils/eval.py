import os
import re
import jieba
import string

import logging
import subprocess
import requests
import wget
import psutil
import time
import json
import socket
import glob
import sys
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

from stanfordcorenlp import StanfordCoreNLP

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from transformers.models.bert.tokenization_bert import BasicTokenizer
from typing import Dict

from .io import get_cache_folder


CORENLP = 'stanford-corenlp-4.5.2'


class MyStanfordCoreNLP(StanfordCoreNLP):
    def __init__(self, path_or_host, port=9000, memory='4g', lang='en', timeout=1500, quiet=True,
                 logging_level=logging.WARNING, auto_download=True):
        
        self.port = port
        self.memory = memory
        self.lang = lang
        self.timeout = timeout
        self.quiet = quiet
        self.logging_level = logging_level

        logging.basicConfig(level=self.logging_level)

        # Check args
        self._check_args()

        if path_or_host.startswith('http'):
            self.url = path_or_host + ':' + str(port)
            self.path_or_host = path_or_host
            logging.info('Using an existing server {}'.format(self.url))
        else:
            self.path_or_host = os.path.join(path_or_host, CORENLP)

            # Check Java
            if not subprocess.call(['java', '-version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) == 0:
                raise RuntimeError('Java not found.')

            # Check if the dir exists
            if not os.path.isdir(self.path_or_host):
                if auto_download:
                    try:
                        logging.info(f'Downloading {CORENLP} to {self.path_or_host} ...')
                        zip, jars = None, []
                        for url in [
                            f'https://nlp.stanford.edu/software/{CORENLP}.zip',
                            f'https://nlp.stanford.edu/software/{CORENLP}-models-german.jar',
                            f'https://nlp.stanford.edu/software/{CORENLP}-models-french.jar',
                        ]:
                            fn = url.split('/')[-1]

                            if fn.endswith('.jar'):
                                jars.append(fn)
                            else:
                                assert zip is None
                                assert fn.endswith('.zip')
                                zip = fn

                            wget.download(url, fn)
                        
                        root = os.path.dirname(self.path_or_host)
                        os.system(f'mkdir -p {root}')
                        os.system(f'unzip {zip} -d {root}')
                        for jar in jars:
                            os.system(f'mv {jar} {os.path.join(root, zip.replace(".zip", ""))}')
                        os.system(f'rm {zip}')
                    except:
                        raise ConnectionError('can not automatically download corenlp-4.5.2')
                else:
                    raise IOError(str(self.path_or_host) + ' is not a directory.')
            directory = os.path.normpath(self.path_or_host) + os.sep
            self.class_path_dir = directory

            # Check if the language specific model file exists
            # We rewrite the file format compaired with the implementation in 3.9.1.1
            switcher = {
                'en': 'stanford-corenlp-[0-9].[0-9].[0-9]-models.jar',
                'zh': 'stanford-corenlp-[0-9].[0-9].[0-9]-models-chinese.jar',
                'ar': 'stanford-corenlp-[0-9].[0-9].[0-9]-models-arabic.jar',
                'fr': 'stanford-corenlp-[0-9].[0-9].[0-9]-models-french.jar',
                'de': 'stanford-corenlp-[0-9].[0-9].[0-9]-models-german.jar',
                'es': 'stanford-corenlp-[0-9].[0-9].[0-9]-models-spanish.jar',
            }
            jars = {
                'en': 'stanford-corenlp-x.x.x-models.jar',
                'zh': 'stanford-corenlp-x.x.x-models-chinese.jar',
                'ar': 'stanford-corenlp-x.x.x-models-arabic.jar',
                'fr': 'stanford-corenlp-x.x.x-models-french.jar',
                'de': 'stanford-corenlp-x.x.x-models-german.jar',
                'es': 'stanford-corenlp-x.x.x-models-spanish.jar',
            }
            if len(glob.glob(directory + switcher.get(self.lang))) <= 0:
                raise IOError(jars.get(
                    self.lang) + ' not exists. You should download and place it in the ' + directory + ' first.')

            # We disable port checking because it will raise an error when running on Mac

            # # If port not set, auto select
            # if self.port is None:
            #     for port_candidate in range(9000, 65535):
            #         if port_candidate not in [conn.laddr[1] for conn in psutil.net_connections()]:
            #             self.port = port_candidate
            #             break

            # # Check if the port is in use
            # if self.port in [conn.laddr[1] for conn in psutil.net_connections()]:
            #     raise IOError('Port ' + str(self.port) + ' is already in use.')

            # Start native server
            logging.info('Initializing native server...')
            cmd = "java"
            java_args = "-Xmx{}".format(self.memory)
            java_class = "edu.stanford.nlp.pipeline.StanfordCoreNLPServer"
            class_path = '"{}*"'.format(directory)

            args = [cmd, java_args, '-cp', class_path, java_class, '-port', str(self.port)]

            args = ' '.join(args)

            logging.info(args)

            # Silence
            with open(os.devnull, 'w') as null_file:
                out_file = None
                if self.quiet:
                    out_file = null_file

                self.p = subprocess.Popen(args, shell=True, stdout=out_file, stderr=subprocess.STDOUT)
                logging.info('Server shell PID: {}'.format(self.p.pid))

            self.url = 'http://localhost:' + str(self.port)

        # Wait until server starts
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_name = urlparse(self.url).hostname
        time.sleep(1)  # OSX, not tested
        while sock.connect_ex((host_name, self.port)):
            logging.info('Waiting until the server is available.')
            time.sleep(1)
        logging.info('The server is available.')

    def _request(self, annotators=None, data=None, *args, **kwargs):
        if sys.version_info.major >= 3:
            data = data.encode('utf-8')

        properties = {'annotators': annotators, 'outputFormat': 'json'}
        params = {'properties': str(properties), 'pipelineLanguage': self.lang}
        if 'pattern' in kwargs:
            params = {"pattern": kwargs['pattern'], 'properties': str(properties), 'pipelineLanguage': self.lang}

        # logging.info(params)
        r = requests.post(self.url, params=params, data=data, headers={'Connection': 'close'})
        r_dict = json.loads(r.text)

        return r_dict



zh_punctuation = "\【.*?】+|\《.*?》+|\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\，。=？、：“”‘’￥……()《》【】～]"


def tokenize_zh_sentence(text:str):
    text = ' '.join(jieba.cut(text))
    text = re.sub(zh_punctuation, '', text).strip()
    text = re.sub(r"\s{2,}", ' ', text)
    return text


def tokenize_sentence(text: str, nlp: MyStanfordCoreNLP=None, punctuation=string.punctuation, do_lower_case=True):
    if nlp is None:
        tokenizer = BasicTokenizer(strip_accents=False, do_lower_case=do_lower_case)
        tokens = tokenizer.tokenize(text)
    else:
        if do_lower_case:
            text = text.lower()
        tokens = [token for token in nlp.word_tokenize(text) if token not in punctuation]
    return ' '.join(tokens)


class MyCOCOEvalCap(COCOEvalCap):
    def __init__(self, coco, cocoRes, lang='zh', cache_folder=get_cache_folder()):
        super().__init__(coco, cocoRes)
        self.lang = lang
        if self.lang != 'zh':
            self.nlp = MyStanfordCoreNLP(cache_folder, lang=lang)

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            if self.lang == 'zh':
                gts[imgId] = [tokenize_zh_sentence(item['caption']) for item in self.coco.imgToAnns[imgId]]
                res[imgId] = [tokenize_zh_sentence(''.join(item['caption'].split(' '))) for item in self.cocoRes.imgToAnns[imgId]]
            else:
                gts[imgId] = [tokenize_sentence(item['caption'], self.nlp) for item in self.coco.imgToAnns[imgId]]
                res[imgId] = [tokenize_sentence(item['caption'], self.nlp) for item in self.cocoRes.imgToAnns[imgId]]

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE") 
            # SPICE will call English-related tokenizer and NER, so we don't evaluate it on other language.
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()


def coco_caption_eval(annotation_file, results_file, eval_lang='en', cache_folder=get_cache_folder()):
    assert os.path.exists(annotation_file)

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    if eval_lang == 'en':
        # we keep using the original evaluation toolkit for visual captioning in English
        coco_eval = COCOEvalCap(coco, coco_result)
    else:
        coco_eval = MyCOCOEvalCap(coco, coco_result, eval_lang, cache_folder=cache_folder)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}', flush=True)

    return coco_eval


def translate_eval(gts, res, eval_lang='en', cache_folder=get_cache_folder()) -> Dict[str, float]:
    assert isinstance(gts, (list, tuple))
    assert isinstance(res, (list, tuple))
    assert len(gts) == len(res)

    if eval_lang == 'zh':
        gts = [[tokenize_zh_sentence(item).split()] for item in gts]
        res = [tokenize_zh_sentence(''.join(item.split())).split() for item in res]
    else:
        nlp = MyStanfordCoreNLP(cache_folder, lang=eval_lang)
        gts = [[tokenize_sentence(item, nlp).split()] for item in gts]
        res = [tokenize_sentence(item, nlp).split() for item in res]

    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    score = corpus_bleu(gts, res, smoothing_function=SmoothingFunction().method1)
    return {
        'BLEU': score,
    }
