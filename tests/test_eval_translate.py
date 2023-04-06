import unittest
from zeronlg.utils import translate_eval


class EvalTranslateTest(unittest.TestCase):
    def test_chinese(self):
        gts = ['一个女孩在舞台上唱歌']
        res = ['一个女孩在唱歌']

        score = translate_eval(gts, res, eval_lang='zh')
        print('zh', score)
    
    # if you run this test for the first time, it may download stanford-corenlp-4.5.2
    def test_english(self):
        gts = ['a girl is singing on the stage']
        res = ['a girl is singing on the stage']

        score = translate_eval(gts, res, eval_lang='en')
        print('en', score)
        assert score['BLEU'] == 1.0
    
    def test_german(self):
        gts = ['ein Mädchen singt auf der Bühne']
        res = ['ein Mädchen singt']

        score = translate_eval(gts, res, eval_lang='de')
        print('de', score)
    
    def test_french(self):
        gts = ['Une fille chante sur scène']
        res = ['Une fille chante']

        score = translate_eval(gts, res, eval_lang='fr')
        print('fr', score)
    

if "__main__" == __name__:
    unittest.main()
