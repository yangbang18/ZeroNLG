import os
import json
import shutil
import unittest

from zeronlg import CaptionDataset, CaptionEvaluator, ZeroNLG
from torch.utils.data import DataLoader
from pathlib import Path
import wget


class EvaluatorCaptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parent.joinpath(self.__class__.__name__)
        os.makedirs(self.root, exist_ok=True)

        self.model = ZeroNLG('zeronlg-4langs-vc')

        wget.download(
            'https://img0.baidu.com/it/u=2179151004,2612321767&fm=253&fmt=auto&app=138&f=JPEG?w=889&h=500', 
            os.path.join(self.root, '0.jpg')
        )

        ann_rpath = os.path.join(self.root, 'ann.json')
        with open(ann_rpath, 'w') as wf:
            json.dump([{'image': '0.jpg'}], wf)
        
        gt_file_path = os.path.join(self.root, 'gts.json')
        with open(gt_file_path, 'w') as wf:
            json.dump(
                {
                    'annotations': [
                        {
                            "image_id": 0,
                            "caption": "two white and cute dogs",
                            "id": -1
                        },
                    ],
                    "images": [
                        {'id': 0},
                    ]
                }, wf
            )
        
        evaluation_settings = {'lang': 'en'}
        
        dataset = CaptionDataset(
            vision_root=self.root,
            ann_rpath=ann_rpath,
            lang=evaluation_settings['lang'],
            clip_model=self.model.clip_model,
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )

        self.evaluator = CaptionEvaluator(
            loader=loader,
            gt_file_path=gt_file_path,
            evaluation_settings=evaluation_settings,
            mode='val'
        )
                
    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def test(self):
        score = self.evaluator(
            model=self.model,
            output_path=self.root,
            epoch=0,
            steps=100,
            print_sent=True
        )
        print(score)


if "__main__" == __name__:
    unittest.main()
