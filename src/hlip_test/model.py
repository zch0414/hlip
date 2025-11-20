import os
import json
import sys
sys.path.append(os.path.abspath('..'))

import torch
from open_clip import create_model_and_transforms
from open_clip.factory import _MODEL_CONFIGS

from hlip import visual_encoder, visual_encoder_rope


if __name__ == '__main__':
    for _c in os.listdir('../hlip/model_configs/'):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join('../hlip/model_configs/', _c), 'r') as f:
                model_cfg = json.load(f)
            _MODEL_CONFIGS[_m] = model_cfg
    model, _, _ = create_model_and_transforms('ablate_seqposemb_clip_vit_base_multiscan_h2_dinotxt1568', output_dict=True, force_patch_dropout=0.50) # replace with test model
    
    x = torch.randn(4, 10, 1, 48, 224, 224)
    
    model_out = model(image=x)
    image_features = model_out['image_features']

    print(image_features.shape)