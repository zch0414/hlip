import os
import sys
sys.path.append(os.path.abspath('..'))

import torch
from timm import create_model

from hlip import visual_encoder, visual_encoder_rope


if __name__ == '__main__':
    model = create_model('ablate_seqposemb_vit_base_multiscan_h2_dinotxt1568', pretrained=True) # replace with the test model
    x = torch.randn(4, 10, 1, 48, 224, 224)
    y = model(x)
    print(y.shape)
