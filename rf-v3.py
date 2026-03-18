import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# Hyperparameters / config
IMAGE_SIZE = (32, 32)
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)
DATA_ROOT = "./data"
BATCH_SIZE = 512
EPOCHS = 5000
AUTOENCODER_EPOCHS = 25
N_WORKERS = 4
MODEL_DIM = 512
MODEL_LAYERS = 8
LEARNING_RATE = 1e-4
CLASSIFIER_FREE_DROP_PROB = 0.5
LOAD_MODEL = False
TRAIN_MODEL = True
CHECKPOINT_PATH = "checkpoints/rf-v3.pth"
AUTOMIXED_DTYPE = torch.bfloat16
MATMUL_PRECISION = "high"
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = N_WORKERS > 0

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ]
)

dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=transform,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
)


class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x_shape = x.size()
        x = x.view(x_shape[0], x_shape[1], -1)
        x = x.permute(0, 2, 1)
        return x


class BlenderInternals(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.prompt_embed = nn.Embedding(11, dim)
        self.prompt_unembed = nn.Linear(dim, 1)
        self.timestep_fc = nn.Linear(dim, dim)

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
        emb = t.unsqueeze(1) * freqs
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
        prompt = prompt + timestep

        image, prompt = self.ln1(image), self.ln1(prompt)
        image_q = self.image_q(image)
        image_k = self.image_k(image)
        image_v = self.image_v(image)

        prompt_q = self.prompt_q(prompt)
        prompt_k = self.prompt_k(prompt)
        prompt_v = self.prompt_v(prompt)

        image_q, image_k, image_v, prompt_q, prompt_k, prompt_v = (
            self.ln1(image_q),
            self.ln1(image_k),
            self.ln1(image_v),
            self.ln1(prompt_q),
            self.ln1(prompt_k),
            self.ln1(prompt_v),
        )

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
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=4,
                               stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim // 2, dim // 4,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim // 4, dim // 8,
                               kernel_size=4, stride=2, padding=1),
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

    def encode(self, image):
        x = self.encoder(image)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, image, prompt, timestep):
        image = self.encode(image)
        prompt = self.blenderhead.embed_prompt(prompt)

        res_img = image
        res_prompt = prompt
        image, prompt = self.blenderhead(image, prompt, timestep)
        image = image + res_img
        prompt = prompt + res_prompt

        for i in range(len(self.blenderinternals)):
            res_img = image
            res_prompt = prompt
            image, prompt = self.blenderinternals[i](image, prompt, timestep)
            image = image + res_img
            prompt = prompt + res_prompt
        return image

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


model = BlenderV2(dim=MODEL_DIM, num_layers=MODEL_LAYERS)
if LOAD_MODEL:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("model load")
model = model.to(device)
torch.set_float32_matmul_precision(MATMUL_PRECISION)
compiled_model = torch.compile(model)
optimizer = optim.Adam(lr=LEARNING_RATE, params=model.parameters())

if TRAIN_MODEL:
    pbar_ae = tqdm(total=len(dataloader) * AUTOENCODER_EPOCHS, ncols=100)

    print("training ae")
    for epoch in range(AUTOENCODER_EPOCHS):
        for count, (image, _) in enumerate(dataloader):
            image = image.to(device, non_blocking=PIN_MEMORY)
            image_original = image
            image = model.encoder(image)
            image = model.decoder(image)

            loss = F.mse_loss(image, image_original)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar_ae.update(1)
            pbar_ae.set_postfix({"loss": f"{loss.item():.4f}", "epoch": epoch})

    pbar_ae.close()
    pbar = tqdm(total=len(dataloader) * EPOCHS, ncols=100)

    print("freezing ae weights")
    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.decoder.parameters():
        param.requires_grad = False

    print("training rf")
    for epoch in range(EPOCHS):
        for count, (image, prompt) in enumerate(dataloader):
            image = image.to(device, non_blocking=PIN_MEMORY)
            prompt = prompt.to(device, non_blocking=PIN_MEMORY)

            if torch.rand(1).item() < CLASSIFIER_FREE_DROP_PROB:
                prompt = torch.full_like(prompt, 10)

            t = torch.rand((image.size(0), 1), device=device)

            with torch.autocast(device_type=device.type, dtype=AUTOMIXED_DTYPE):
                x1 = compiled_model.encoder(image)
                x0 = torch.randn_like(x1)

                t_broadcast = t.unsqueeze(-1)
                x_t = x0 * (1 - t_broadcast) + x1 * t_broadcast
                v_predicted = compiled_model.forward_latent(x_t, prompt, t)

                v_target = x1 - x0
                loss = F.mse_loss(v_predicted, v_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "epoch": epoch})

    pbar.close()

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"saved model weights to {CHECKPOINT_PATH}")
