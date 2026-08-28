"""Class-conditional LoRA fine-tuning of Stable Diffusion 1.5 for mammograms.

SOTA-generator track (Phase 2). Instead of training a GAN from scratch, we
LoRA-fine-tune a pretrained latent-diffusion UNet on the DEV images only
(leakage-safe), conditioned on simple clinical prompts per class. LoRA trains
<1% of the parameters, so it fits a consumer GPU (RTX 4060, 8 GB) with fp16 +
gradient checkpointing — no A100 required.

Domain-gap caveat: SD 1.5 is pretrained on natural RGB images; mammograms are
grayscale and out-of-distribution, so LoRA adaptation on a few thousand images is
expected to help but may not reach photographic fidelity. Memorisation is audited
downstream (diffusion + few-shot LoRA can replicate training images).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
PROMPTS = {
    0: "a screening mammogram showing benign breast tissue, grayscale, high detail",
    1: "a screening mammogram showing a malignant breast mass, grayscale, high detail",
}


class _PromptImageDataset(Dataset):
    def __init__(self, paths, labels, size=512):
        self.paths, self.labels = paths, labels
        self.t = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # -> [-1,1], VAE expects this
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")  # grayscale -> 3ch for SD VAE
        return self.t(img), PROMPTS[int(self.labels[i])]


def train_lora(
    paths, labels, out_dir, *, size=512, rank=8, steps=1500, lr=1e-4,
    batch_size=1, grad_accum=4, device="cuda", seed=0,
):
    """LoRA-fine-tune the SD1.5 UNet on (paths, labels). Saves adapter to out_dir."""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16

    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder").to(device, dtype)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae").to(device, dtype)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet").to(device)
    noise_sched = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    lora = LoraConfig(
        r=rank, lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora)
    unet.to(device)
    unet.train()
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    ds = _PromptImageDataset(paths, labels, size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    def encode_text(prompts):
        ids = tokenizer(list(prompts), padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            return text_encoder(ids)[0]

    step, done = 0, False
    pbar = tqdm(total=steps, desc="LoRA-SD")
    while not done:
        for imgs, prompts in loader:
            imgs = imgs.to(device, dtype)
            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy = noise_sched.add_noise(latents, noise, t)
            enc = encode_text(prompts)
            with torch.autocast("cuda", dtype=dtype):
                pred = unet(noisy, t, encoder_hidden_states=enc).sample
                target = noise if noise_sched.config.prediction_type == "epsilon" else \
                    noise_sched.get_velocity(latents, noise, t)
                loss = F.mse_loss(pred.float(), target.float()) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1); pbar.set_postfix(loss=f"{loss.item() * grad_accum:.3f}")
            if step % 500 == 0 or step >= steps:
                unet.save_pretrained(out_dir)
            if step >= steps:
                done = True
                break
    pbar.close()
    unet.save_pretrained(out_dir)
    return out_dir


@torch.no_grad()
def generate_lora(lora_dir, class_label, n, out_dir, *, size=512, steps=30,
                  guidance=4.0, device="cuda", seed=0):
    """Generate n synthetic images for a class from the LoRA-adapted SD1.5."""
    from diffusers import StableDiffusionPipeline
    from peft import PeftModel

    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16,
                                                   safety_checker=None).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_dir)).to(device, torch.float16)
    pipe.set_progress_bar_config(disable=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device=device).manual_seed(seed + class_label * 100000)
    name = {0: "BENIGN", 1: "MALIGNANT"}[class_label]
    saved = []
    for i in tqdm(range(n), desc=f"gen {name}"):
        img = pipe(PROMPTS[class_label], num_inference_steps=steps, guidance_scale=guidance,
                   height=size, width=size, generator=g).images[0]
        p = out_dir / f"{name}_{i + 1:04d}.png"
        img.convert("L").save(p)
        saved.append(str(p))
    return saved
