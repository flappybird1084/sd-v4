"""Gradio app for sampling MNIST digits from the rf-v3 rectified-flow model.

Pick a digit (0-9), hit generate, and the model integrates the latent ODE from
noise with classifier-free guidance and decodes a 32x32 image.

    uv run app.py
    uv run app.py --checkpoint checkpoints/rf-v3-demo.pth
"""

import argparse
import math

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model config (must match the trained checkpoint).
MODEL_DIM = 512
MODEL_LAYERS = 8
LATENT_TOKENS = 16
NULL_TOKEN = 10  # classifier-free-guidance null prompt index

# Sampling defaults (from rf-v3.ipynb).
NUM_STEPS = 50
GUIDANCE_SCALE = 5.0

# Upscale factor for display (32x32 -> 32*DISPLAY_SCALE square).
DISPLAY_SCALE = 10


class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv(x)
        x_shape = x.size()
        x = x.view(x_shape[0], x_shape[1], -1)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x


class BlenderInternals(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.prompt_embed = nn.Embedding(11, dim)
        self.prompt_unembed = nn.Linear(dim, 1)
        self.timestep_fc = nn.Linear(dim, dim)
        self.image_embed = nn.Embedding(16, dim)
        self.register_buffer("pos_ids", torch.arange(16))

        self.image_q = nn.Linear(dim, dim)
        self.image_v = nn.Linear(dim, dim)
        self.image_k = nn.Linear(dim, dim)

        self.prompt_q = nn.Linear(dim, dim)
        self.prompt_v = nn.Linear(dim, dim)
        self.prompt_k = nn.Linear(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        half_dim = dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim)
        self.register_buffer("timestep_freq", freq)

    def timestep_embedding(self, t):
        if t.dim() == 2:
            t = t.squeeze(-1)
        freqs = self.timestep_freq.unsqueeze(0)
        emb = t.unsqueeze(1) * freqs * 1000
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.timestep_fc(emb)

    def embed_prompt(self, prompt):
        prompt = prompt.unsqueeze(-1)
        prompt = self.prompt_embed(prompt)
        return prompt

    def forward(self, image, prompt, timestep):
        timestep = self.timestep_embedding(timestep)
        timestep = timestep.unsqueeze(1)

        image = image + timestep
        image = torch.add(image, self.image_embed(self.pos_ids))
        prompt = prompt + timestep

        image_norm, prompt_norm = self.ln1(image), self.ln1(prompt)
        image_q = self.image_q(image_norm)
        image_k = self.image_k(image_norm)
        image_v = self.image_v(image_norm)

        prompt_q = self.prompt_q(prompt_norm)
        prompt_k = self.prompt_k(prompt_norm)
        prompt_v = self.prompt_v(prompt_norm)

        q = torch.cat([image_q, prompt_q], dim=1)
        k = torch.cat([image_k, prompt_k], dim=1)
        v = torch.cat([image_v, prompt_v], dim=1)

        out = torch.cat([image, prompt], dim=1)
        out = out + F.scaled_dot_product_attention(q, k, v)
        out = self.ln2(out)
        out = out + self.mlp(out)
        out_image = out[:, :-1, :]
        out_prompt = out[:, -1, :]

        out_prompt = out_prompt.unsqueeze(1)
        return out_image, out_prompt


class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        h = w = int(math.sqrt(seq_len))
        assert h * w == seq_len, "L must be a perfect square"
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(bsz, dim, h, w)
        x = self.decoder(x)
        return x


class BlenderV2(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.encoder = Encoder(dim)
        self.blenderhead = BlenderInternals(dim)
        self.blenderinternals = nn.ModuleList(
            [BlenderInternals(dim) for _ in range(num_layers - 1)]
        )
        self.decoder = Decoder(dim)

    def forward_latent(self, z, prompt, t):
        prompt = self.blenderhead.embed_prompt(prompt)

        res_img = z
        res_prompt = prompt
        z, prompt = self.blenderhead(z, prompt, t)
        z = z + res_img
        prompt = prompt + res_prompt

        for block in self.blenderinternals:
            res_img = z
            res_prompt = prompt
            z, prompt = block(z, prompt, t)
            z = z + res_img
            prompt = prompt + res_prompt
        return z

    def decode(self, x):
        return self.decoder(x)


def load_model(checkpoint_path, device):
    model = BlenderV2(dim=MODEL_DIM, num_layers=MODEL_LAYERS)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(model, device, digit, num_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE, seed=None):
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    else:
        generator = None

    prompt = torch.tensor([int(digit)], device=device)
    null = torch.full_like(prompt, NULL_TOKEN)

    x = torch.randn(1, LATENT_TOKENS, MODEL_DIM, device=device, generator=generator)
    dt = 1.0 / num_steps

    use_autocast = device.type == "cuda"
    for step in range(num_steps):
        t = torch.ones((1, 1), device=device) * ((step + 0.5) / num_steps)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
            v_cond = model.forward_latent(x, prompt, t)
            v_uncond = model.forward_latent(x, null, t)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        x = x + v * dt

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
        img = model.decode(x)

    img = (img.clamp(-1, 1) + 1) / 2
    img = img[0, 0].float().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    # Upscale with nearest-neighbor so the 32x32 output fills the display box crisply.
    img = np.kron(img, np.ones((DISPLAY_SCALE, DISPLAY_SCALE), dtype=np.uint8))
    return img


def build_interface(model, device):
    def fn(digit, num_steps, guidance_scale, seed):
        seed = None if seed is None or int(seed) < 0 else int(seed)
        return generate(
            model,
            device,
            int(digit),
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            seed=seed,
        )

    with gr.Blocks(title="rf-v3 — MNIST digit generator") as demo:
        gr.Markdown(
            "# rf-v3 — MNIST digit generator\n"
            "Pick a digit and generate an image with the rectified-flow model."
        )
        with gr.Row():
            with gr.Column():
                digit = gr.Dropdown(
                    choices=[str(i) for i in range(10)],
                    value="4",
                    label="Digit prompt",
                )
                num_steps = gr.Slider(1, 200, value=NUM_STEPS, step=1, label="Sampling steps")
                guidance_scale = gr.Slider(
                    0.0, 15.0, value=GUIDANCE_SCALE, step=0.5, label="Guidance scale"
                )
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out = gr.Image(label="Generated", image_mode="L", height=320, width=320)

        btn.click(fn, inputs=[digit, num_steps, guidance_scale, seed], outputs=out)
    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio app for rf-v3 digit generation")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/rf-v3.pth",
        help="Path to model checkpoint (overrides the default).",
    )
    parser.add_argument(
        "--no-share", dest="share", action="store_false", help="Disable the public Gradio link."
    )
    parser.set_defaults(share=True)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.checkpoint} on {device}")
    model = load_model(args.checkpoint, device)

    demo = build_interface(model, device)
    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
