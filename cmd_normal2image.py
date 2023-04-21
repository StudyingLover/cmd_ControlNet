from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random
import argparse

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_midas = MidasDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_normal.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, bg_threshold):
    with torch.no_grad():
        input_image = HWC3(input_image)
        _, detected_map = apply_midas(resize_image(input_image, detect_resolution), bg_th=bg_threshold)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map[:, :, ::-1].copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


if '__main__' == __name__:

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_test', type=bool, default=False, help='run a easy test')
    parser.add_argument('--image_path', type=str, default='test.png', help='original image path')
    parser.add_argument('--prompt', type=str, default='1people', help='prompt')
    parser.add_argument('--a_prompt', type=str, default='best quality, extremely detailed', help='added prompt')
    parser.add_argument('--n_prompt', type=str, default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help='negative prompt')
    parser.add_argument('--num_samples', type=int, default=1, help='number of samples')
    parser.add_argument('--image_resolution', type=int, default=512, help='image resolution')
    parser.add_argument('--detect_resolution', type=int, default=512, help='detect resolution')
    parser.add_argument('--ddim_steps', type=int, default=30, help='ddim steps')
    parser.add_argument('--is_saved', type=bool, default=True, help='is saved?')
    parser.add_argument('--is_show', type=bool, default=False, help='is show?')
    parser.add_argument('--guess_mode', type=bool, default=False, help='guess mode')
    parser.add_argument('--strength', type=float, default=1.0, help='strength')
    parser.add_argument('--scale', type=float, default=9.0, help='scale')
    parser.add_argument('--seed', type=int, default=-1, help='seed')
    parser.add_argument('--eta', type=float, default=0.0, help='eta')
    parser.add_argument('--bg_threshold', type=float, default=0.1, help='bg_threshold')


    opt = parser.parse_args()

    img=cv2.imread(opt.image_path)
    out=process(img, opt.prompt, opt.a_prompt, opt.n_prompt, opt.num_samples, opt.image_resolution, opt.detect_resolution, opt.ddim_steps, opt.guess_mode, opt.strength, opt.scale, opt.seed, opt.eta, opt.bg_threshold)

    if(opt.is_show):
        cv2.imshow('out',out[1])
    if(opt.is_saved):
        cv2.imwrite('out.png',out[1])
        print('saved to out.png')