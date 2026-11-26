import unittest
import numpy as np
import torch
from PIL import Image
from src.vision import pinhole,filters,edges,frequency,features,match_views
from src.model import FruitCNN,predict

class Tests(unittest.TestCase):
    def test_projection(self):
        np.testing.assert_allclose(pinhole([[2,4,2]],np.eye(3)),[[1,2]])
    def test_pipeline(self):
        img=np.random.default_rng(42).integers(0,256,(120,150,3),dtype=np.uint8)
        self.assertEqual(filters(img)['Median'].shape,img.shape)
        self.assertEqual(edges(img)['Canny'].shape,img.shape[:2]);self.assertEqual(frequency(img)['Low pass'].shape,img.shape[:2]);self.assertEqual(len(features(img)[1]),3)
        self.assertGreater(match_views(img,img)[1]['ratio_matches'],0)
    def test_model_shape(self):
        model=FruitCNN(); self.assertEqual(tuple(model(torch.zeros(2,3,128,128)).shape),(2,2))
        self.assertAlmostEqual(sum(predict(model,Image.new('RGB',(128,128))).values()),1,places=5)
if __name__=='__main__':unittest.main()
