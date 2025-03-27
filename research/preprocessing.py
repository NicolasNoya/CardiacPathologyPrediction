#%%https://pdf.sciencedirectassets.com/271322/1-s2.0-S0169260722X00072/1-s2.0-S0169260722002978/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEK3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIBZFiin5aBIv6thGFV5fTVrbuWONfyE9nQJaZvXK47ElAiEAupnX4FYggcZRjHR4IVuEDw37lDnIxkR121X3a25EzBIqsgUIFRAFGgwwNTkwMDM1NDY4NjUiDJTxg8i9F%2BKOIwGdyyqPBcGMBCxWVdSswPVYHTJE2H1Wp7Y6urw1SLBNTNudm%2Fmj5qTSHpAVu%2BdgC3lxbxNLXAmlB9CDSYe3aecvMKOffgolwpBUP5%2Fhd5gVUY66xmW%2BrG0orB0yLEu9CyeBSC%2FerShzaRjV5T1jHyuSZGQddQoH512uvXZBgRkkI1LxrAifUoMYJ7V2BUsQevjDQmM5wfwV%2FBEuQAIXTfLUEd8OlM4Fgd3zieCsVyTV9lGbYRM3mCuIqAxX2xRFxe79P6bTbCCOcZqjdofokZICen6fHHeWU5KQnQyoYObadiwCm2BE4gekX2CCiLYnWUZ4e5sNcPPjTvkt2vezvqjeCAlsD%2B9p5B0U8miFxpgIM%2BHWcftD6sJRi7HJIvAYK1Bu6nW7cRNan8LfdqqrvBJn74nrezLypzFi7voimQvrk8v1AjUXdcujieSGKP1ry%2BjxASD6aQPmAu7IwjBRQcNSPiKIDrOkJeWmqbOtSEMgID0WNYx0wudfrKtmmY27JrmmsClGVLMzaCzom0MsdUI6k9djyUSmbooJRAtvS9Hl53opVc2vS4WzkM5e0%2BASFKwCs%2Fy3O0lpSCKhStHHpfS864zHcTQdXFXpyurQ0Bbqm9ygHWPel%2BvfFgE55n%2BV3LhAnDhambwAwfWlzexJidcWfCAwlNZJZqE7HJad%2BfD7AwpXtXzv8CAEFLaySP2P0Tk4z3d6kwvt6FrvtSr5kkPmCd5iyJXoteyUuQxe5A2sH5BhjImpaHQnhsvjaDoP28Qyb2CCATGueploke14%2BEYrii3fwlXdQqAOAuKyxQ4lqMOPWzIzAKRFpmKWWNHRUBxbPq4n9wTH6s73Sc0zgDu%2FBEgJ7hzO8e%2FGC8SXdH6ev4gPmGcwgLiKvwY6sQGeastj2CSqYEpHMFrqGmM%2FeJikSPGizwkEaXxRhdJMwsZf%2FmL9uZpeelVvapwH3wOMVrgBHw3vXRvY%2BJ8by0UtsuH%2FqaLDzLHrN8ACm%2FLB3D2OZojjO1cFTpGmNSV1MJqUy9oXTwxk3jlL4gaYkksi7BS40nw929ym%2FDfcm9TEd%2BO24OtNY1%2FVMsSXIc8QZ86TMb5Z0XDifE2nhuemtDGVhVnF8tS%2BGpRjJKBCa%2BcvyXQ%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250325T130247Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYWDBDO3KR%2F20250325%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=1156784a68ac5d9d1245ffd8ea58d9dec83a3ad23ccd487a6f1c8557bbe44b9b&hash=4ee470563a03d3352a185ad5b5d49db029773ba9bb7e6ff3dab0b32cb33490c3&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0169260722002978&tid=spdf-450a9723-b012-4b17-96e2-da913f7cfbf1&sid=9db4a0ae58d5f848ba3b0e769a2dd20d7580gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0014565354065603040c&rr=925e9acb1d6fd15e&cc=fr
import torchvision.transforms.functional as F1
from niidataloader import NiftiDataset
import matplotlib.pyplot as plt
import torch

class Preprocessing:
    def __init__(self):
        pass

    def preprocess(self, img: torch.Tensor):
        #blur the image with a gaussian filter
        img = img.to(torch.float32)
        img=img.reshape(1,220,220)
        image_blur = F1.gaussian_blur(img, kernel_size=3, sigma=2.0)
        image_blur1 = F1.gaussian_blur(image_blur, kernel_size=5, sigma=2.0)
        # Add some noise
        image_noise = torch.randn_like(image_blur1) * 0.03 # this is to reduce the variance of the noise
        final_image = image_blur1 + image_noise
        return  final_image