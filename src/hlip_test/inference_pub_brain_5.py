import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from open_clip import create_model_and_transforms, get_tokenizer, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load

import torch
from torchvision.transforms import Normalize

from timm.models.vision_transformer import Attention, Block
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from hlip import visual_encoder
from hlip.zeroshot_metadata_pub_brain_5 import PROMPTS, TEMPLATES


def get_args_parser():
    parser = argparse.ArgumentParser('Inference', add_help=False)
    
    # model
    parser.add_argument('--model', default='clip_vit_base_multiscan_h2_token588', type=str)
    parser.add_argument('--patch-size', nargs='+', default=[16, 16, 16], type=int)
    parser.add_argument('--resume', default='/pretrained/brainmri_clip_vit_base_multiscan_h2_token588.pt', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    
    # data
    parser.add_argument('--data', default='../../docs/BraTS-GLI-00459-000/')
    parser.add_argument('--num-slices', default=144, type=int)

    # interpret
    parser.add_argument('--interpret', default=False, action='store_true')
    parser.add_argument('--target', default='Glioma', type=str)
    parser.add_argument('--save-dir', default='./results/', type=str)
    return parser


def loader(study_path, num_slices):
    imgs = []
    for scan in [os.path.join(study_path, p) for p in os.listdir(study_path)]:
        # load image
        img = torch.load(scan, weights_only=True)
        if len(img.shape) == 4:
            img = img[:, :, :, 0]
        img = img[None, ...].float()

        # padding to the longest side   
        _, _, h, w = img.shape             
        size = max(h, w)
        pad_h = size - h; pad_w = size - w
        left = pad_w // 2; right = pad_w - left; top = pad_h // 2; bottom = pad_h - top
        img = torch.nn.functional.pad(img, (left, right, top, bottom), mode="constant", value=0)

        # resize to 256, crop to 224
        img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bilinear')
        img = torch.nn.functional.interpolate(img[None, ...], size=(num_slices, 256, 256), mode='nearest-exact').squeeze(0)
        img = img[:, :, 16:240, 16:240]

        # normalize
        normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
        img = normalizer(img)
        imgs.append(img)

    return torch.stack(imgs, dim=0)[None, ...] # [1, n, 1, d, h, w]


def show_interpret(image, interpret, save_name, args):
    save_dir = os.path.join(args.save_dir, args.target)
    os.makedirs(save_dir, exist_ok=True)

    # de-normalization
    image = image * torch.as_tensor(IMAGENET_DEFAULT_STD).mean() + torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean()

    # visualization
    n = image.shape[0]
    m = image.shape[1] // args.patch_size[0]
    fig, axes = plt.subplots(m, n, figsize=(4*n, 4*m))
    if n == 1 and m == 1:
        axes = [axes]
    
    for i in range(n):
        for j, slice_index in enumerate(((np.arange(m) + 0.5) * args.patch_size[0]).astype(int).tolist()):
            image_show = image[i, slice_index, :, :]
            interpret_show = interpret[i, slice_index, :, :]
            interpret_show = (interpret_show - interpret_show.min()) / (interpret_show.max() - interpret_show.min())
            interpret_show[interpret_show < 0.25] = 0
            interpret_show = np.ma.masked_where(interpret_show == 0, interpret_show)

            axes[j, i].imshow(image_show, cmap='gray')
            axes[j, i].imshow(interpret_show, cmap='jet', alpha=0.3)
            axes[j, i].set_title(f"scan {i}, slice {slice_index}")
            axes[j, i].axis('off')

    plt.tight_layout()    
    fig.savefig(os.path.join(save_dir, save_name), dpi=300)


class InterpretAttention(Attention):
    def forward(self, x, forward_hook=None, backward_hook=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if forward_hook is not None:
            forward_hook(attn)
        if backward_hook is not None:
            attn.register_hook(backward_hook)

        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class InterpretBlock(Block):
    def forward_hook(self, attn_prob):
        setattr(self, "attn_prob", attn_prob)

    def backward_hook(self, attn_grad):
        setattr(self, "attn_grad", attn_grad)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.forward_hook, self.backward_hook)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def apply_interpret(model):
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = InterpretBlock
        elif isinstance(module, Attention):
            module.__class__ = InterpretAttention


def inference(model, tokenizer, image, args):
    model.eval()
    device = torch.device(args.device)
    image = image.float().to(device=device)

    # build text embeddings
    classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=PROMPTS['prompt'],
            templates=TEMPLATES['template'],
            num_classes_per_batch=None, # all
            device=device,
            use_tqdm=False,
        )
    
    # warpper model
    apply_interpret(model.visual.trunk)
    
    # inference
    output = model(image=image)
    logit_scale = output['logit_scale']
    image_features = output['image_features']
    logits_per_image = logit_scale * image_features @ classifier
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    for i, prompt in enumerate(PROMPTS['prompt']):
        print(f'{prompt}: {probs[0, i]:.4f}')

    # interpret
    if args.interpret:
        one_hot = np.zeros((1, len(PROMPTS['prompt'])), dtype=np.float32)
        one_hot[0, PROMPTS['prompt'].index(args.target)] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device=device)
        one_hot = torch.sum(one_hot * logits_per_image)
        model.zero_grad()
        one_hot.backward(retain_graph=True)

        num_scans = image.shape[1]
        grid_size = (
            int(image.shape[3] / args.patch_size[0]),
            int(image.shape[4] / args.patch_size[1]),
            int(image.shape[5] / args.patch_size[2])
        )
        num_tokens = num_scans * grid_size[0] * grid_size[1] * grid_size[2] + model.visual.trunk.num_prefix_tokens
    
        R = torch.eye(num_tokens, num_tokens).float().to(device=device)
        grad = model.visual.trunk.blocks[-1].attn_grad
        cam = model.visual.trunk.blocks[-1].attn_prob
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R = R + torch.matmul(cam, R)

        image_global_relevance = R[0, 1:].reshape(num_scans, grid_size[0], grid_size[1], grid_size[2]).detach().cpu()
        image_global_relevance = torch.nn.functional.interpolate(image_global_relevance[None, ...], size=(image.shape[3], image.shape[4], image.shape[5]), mode='trilinear').squeeze()
        save_name = f"{args.data.split('/')[-2]}.png"
        show_interpret(image.cpu().squeeze(), image_global_relevance, save_name, args)
    return


def main(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # create model
    for _c in os.listdir('../hlip/model_configs/'):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join('../hlip/model_configs/', _c), 'r') as f:
                model_cfg = json.load(f)
            _MODEL_CONFIGS[_m] = model_cfg
    model, _, _ = create_model_and_transforms(args.model, device=args.device, output_dict=True)
    checkpoint = pt_load(args.resume, map_location='cpu')
    sd = checkpoint['state_dict']
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    tokenizer = get_tokenizer(args.model)

    # inference
    image = loader(args.data, args.num_slices)
    inference(model, tokenizer, image, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)