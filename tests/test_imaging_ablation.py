import unittest
import numpy as np
import torch
from src.imaging_ablation import dominant_period, folded_channels, gasf_channels, render_image, REPRESENTATIONS

class ImagingMath(unittest.TestCase):
    def test_period_and_fold_order(self):
        x=np.sin(2*np.pi*np.arange(64)/8)[None,:].repeat(3,axis=0)
        self.assertEqual(dominant_period(x),8)
        matrix,mask,period=folded_channels(x)
        np.testing.assert_array_equal(matrix.reshape(3,-1),x)
        self.assertTrue(mask.all())
        self.assertEqual(matrix.shape,(3,8,8))

    def test_gasf_matches_angle_definition(self):
        x=np.array([[-3.,1.,4.,9.],[1.,1.,1.,1.]])
        z=np.array([[-1.,-1/3.,1/6.,1.],[0.,0.,0.,0.]])
        angle=np.arccos(z)
        np.testing.assert_allclose(gasf_channels(x),np.cos(angle[:,:,None]+angle[:,None,:]),atol=1e-7)
        np.testing.assert_allclose(gasf_channels(x),gasf_channels(x).transpose(0,2,1))

    def test_padding_never_enters_images(self):
        torch.manual_seed(7); x=torch.randn(5,57)
        bad=torch.cat([x,torch.full((5,7),float('nan'))],dim=1)
        for mode in REPRESENTATIONS:
            a=render_image(x,image_representation=mode)
            b=render_image(bad,valid_length=57,image_representation=mode)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(tuple(a.shape),(3,224,224))
            self.assertTrue(torch.isfinite(a).all())

    def test_constant_and_singleton(self):
        for n in [1,2,64]:
            self.assertEqual(dominant_period(np.ones((2,n))),n)
            for mode in REPRESENTATIONS:
                image=render_image(torch.ones(2,n),image_representation=mode)
                self.assertTrue(torch.isfinite(image).all())

    def test_distinct_modes(self):
        torch.manual_seed(8); x=torch.randn(4,64)
        images=[render_image(x,image_representation=m) for m in REPRESENTATIONS]
        self.assertTrue(all(not torch.equal(images[i],images[j]) for i in range(3) for j in range(i)))

if __name__=='__main__': unittest.main()
