import os
import json
import shutil
import unittest
from zeronlg.utils import coco_caption_eval
from pathlib import Path


class EvalCaptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parent.joinpath(self.__class__.__name__)
        os.makedirs(self.root, exist_ok=True)
                
    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    # if you run this test for the first time, it may download stanford-corenlp-3.6.0 required by the SPICE metric
    def test_english(self):
        annotation_file = os.path.join(self.root, 'gts.json')
        with open(annotation_file, 'w') as wf:
            json.dump(
                {
                    'annotations': [
                        {
                            "image_id": 100,
                            "caption": "a girl is singing on the stage",
                            "id": -1
                        },
                        {
                            "image_id": 100,
                            "caption": "a beautiful girl is showing her talents",
                            "id": -1
                        },
                        {
                            "image_id": 101,
                            "caption": "a boy is running on the road and wearing a red bag",
                            "id": -1,
                        },
                        {
                            "image_id": 101,
                            "caption": "a black boy looks happy and runs",
                            "id": -1,
                        },
                    ],
                    "images": [
                        {'id': 100},
                        {'id': 101}
                    ]
                }, wf
            )

        results_file = os.path.join(self.root, 'res.json')
        with open(results_file, 'w') as wf:
            json.dump([
                {'image_id': 100, 'caption': 'a beautiful girl is singing'},
                {'image_id': 101, 'caption': 'a handsome boy is running'}
            ], wf)

        coco_eval = coco_caption_eval(annotation_file, results_file, eval_lang='en')
        print('en', coco_eval.eval)
    
    def test_chinese(self):
        annotation_file = os.path.join(self.root, 'gts.json')
        with open(annotation_file, 'w') as wf:
            json.dump(
                {
                    'annotations': [
                        {
                            "image_id": 100,
                            "caption": "一个女孩在舞台上唱歌",
                            "id": -1
                        },
                        {
                            "image_id": 100,
                            "caption": "一个漂亮的女孩在展示她的才华",
                            "id": -1
                        },
                        {
                            "image_id": 101,
                            "caption": "一个穿着红色书包的男孩在路上跑",
                            "id": -1,
                        },
                        {
                            "image_id": 101,
                            "caption": "一个黑皮肤男孩看起来很开心并在奔跑",
                            "id": -1,
                        },
                    ],
                    "images": [
                        {'id': 100},
                        {'id': 101}
                    ]
                }, wf
            )

        results_file = os.path.join(self.root, 'res.json')
        with open(results_file, 'w') as wf:
            json.dump([
                {'image_id': 100, 'caption': '一个漂亮的女孩在唱歌'},
                {'image_id': 101, 'caption': '一个帅气的男孩在奔跑'}
            ], wf)

        coco_eval = coco_caption_eval(annotation_file, results_file, eval_lang='zh')
        print('zh', coco_eval.eval)
    
    def test_german(self):
        annotation_file = os.path.join(self.root, 'gts.json')
        with open(annotation_file, 'w') as wf:
            json.dump(
                {
                    'annotations': [
                        {
                            "image_id": 100,
                            "caption": "Ein Mädchen singt auf der Bühne",
                            "id": -1
                        },
                        {
                            "image_id": 100,
                            "caption": "Ein schönes Mädchen zeigt ihre Talente",
                            "id": -1
                        },
                        {
                            "image_id": 101,
                            "caption": "Ein Junge in einer roten Schultasche lief auf der Straße",
                            "id": -1,
                        },
                        {
                            "image_id": 101,
                            "caption": "Ein schwarzhäutiger Junge sieht glücklich aus und läuft",
                            "id": -1,
                        },
                    ],
                    "images": [
                        {'id': 100},
                        {'id': 101}
                    ]
                }, wf
            )

        results_file = os.path.join(self.root, 'res.json')
        with open(results_file, 'w') as wf:
            json.dump([
                {'image_id': 100, 'caption': 'Ein schönes Mädchen singt'},
                {'image_id': 101, 'caption': 'Ein hübscher Junge rennt'}
            ], wf)

        coco_eval = coco_caption_eval(annotation_file, results_file, eval_lang='de')
        print('de', coco_eval.eval)

    def test_french(self):
        annotation_file = os.path.join(self.root, 'gts.json')
        with open(annotation_file, 'w') as wf:
            json.dump(
                {
                    'annotations': [
                        {
                            "image_id": 100,
                            "caption": "Une fille chante sur scène",
                            "id": -1
                        },
                        {
                            "image_id": 100,
                            "caption": "Une belle fille montre son talent",
                            "id": -1
                        },
                        {
                            "image_id": 101,
                            "caption": "Garçon dans un cartable rouge courant sur la route",
                            "id": -1,
                        },
                        {
                            "image_id": 101,
                            "caption": "Un garçon à la peau noire semble heureux et court",
                            "id": -1,
                        },
                    ],
                    "images": [
                        {'id': 100},
                        {'id': 101}
                    ]
                }, wf
            )

        results_file = os.path.join(self.root, 'res.json')
        with open(results_file, 'w') as wf:
            json.dump([
                {'image_id': 100, 'caption': 'Une belle fille chantant'},
                {'image_id': 101, 'caption': 'Un beau garçon qui court'}
            ], wf)

        coco_eval = coco_caption_eval(annotation_file, results_file, eval_lang='fr')
        print('fr', coco_eval.eval)


if "__main__" == __name__:
    unittest.main()
