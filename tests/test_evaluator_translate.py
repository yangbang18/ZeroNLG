import os
import shutil
import unittest

from zeronlg import TranslateDataset, TranslateEvaluator, ZeroNLG
from torch.utils.data import DataLoader
from pathlib import Path


class EvaluatorTranslateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parent.joinpath(self.__class__.__name__)
        os.makedirs(self.root, exist_ok=True)
        self.model = ZeroNLG('zeronlg-4langs-vc')
                
    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def test_en2zh(self):
        evaluation_settings = {'lang': 'zh'}
        
        dataset = TranslateDataset(
            source_language='en',
            target_language='zh',
            source_sentences=['a girl is singing', 'a boy is running'],
            target_sentences=['一个女孩在唱歌', '一个男孩在奔跑'],
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
        )

        evaluator = TranslateEvaluator(
            loader=loader,
            evaluation_settings=evaluation_settings,
            mode='val'
        )

        score = evaluator(
            model=self.model,
            output_path=self.root,
            epoch=0,
            steps=100,
            print_sent=True
        )
        print(score)


        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )
        evaluator = TranslateEvaluator(
            loader=loader,
            evaluation_settings=evaluation_settings,
            mode='val'
        )
        score = evaluator(
            model=self.model,
            output_path=self.root,
            epoch=0,
            steps=100,
            print_sent=True
        )
        print(score)

    def test_zh2en(self):
        evaluation_settings = {'lang': 'en'}
        
        dataset = TranslateDataset(
            source_language='zh',
            target_language='en',
            source_sentences=['一个女孩在唱歌', '一个男孩在奔跑'],
            target_sentences=['a girl is singing', 'a boy is running'],
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
        )

        evaluator = TranslateEvaluator(
            loader=loader,
            evaluation_settings=evaluation_settings,
            mode='val'
        )

        score = evaluator(
            model=self.model,
            output_path=self.root,
            epoch=0,
            steps=100,
            print_sent=True
        )
        print(score)


if "__main__" == __name__:
    unittest.main()
