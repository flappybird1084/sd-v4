import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import math
# Hyperparameters / config
IMAGE_SIZE = (32, 32)
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)
DATA_ROOT = "./data"
BATCH_SIZE = 512
EPOCHS = 5000
AUTOENCODER_EPOCHS = 10
N_WORKERS = 16
MODEL_DIM = 512
MODEL_LAYERS = 8
LEARNING_RATE = 1e-4
LOAD_MODEL = False
TRAIN_MODEL = True
CHECKPOINT_PATH = "checkpoints/rf-v2.pth"
AUTOMIXED_DTYPE = torch.bfloat16
MATMUL_PRECISION = "high"
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = N_WORKERS > 0
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])
dataset = datasets.MNIST(root=DATA_ROOT, train=True,
                         download=True, transform=transform)
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
            nn.MaxPool2d(2, 2),  # B,32,64,64,
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # B,64,32,32
            nn.ReLU(),
            nn.Conv2d(64, dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # B,dim,16,16
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.conv(x)  # B,dim,16,16
        x_shape = x.size()
        x = x.view(x_shape[0], x_shape[1], -1)  # B,dim,256
        x = x.permute(0, 2, 1)  # B,256,dim
        return x
class BlenderInternals(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.prompt_embed = nn.Embedding(10, dim)  # B,dim
        self.prompt_unembed = nn.Linear(dim, 1)
        self.timestep_fc = nn.Linear(dim, dim)
        self.image_q = nn.Linear(dim, dim)
        self.image_v = nn.Linear(dim, dim)
        self.image_k = nn.Linear(dim, dim)
        self.prompt_q = nn.Linear(dim, dim)
        self.prompt_v = nn.Linear(dim, dim)
        self.prompt_k = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        half_dim = self.dim // 2
        inv_freq = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) *
            (-(math.log(10000.0) / half_dim))
        )
        self.register_buffer("timestep_inv_freq", inv_freq, persistent=False)
    def sinusoidal_timestep_embedding(self, t):
        t = t.view(t.size(0), 1).to(dtype=self.timestep_inv_freq.dtype)
        emb = t * self.timestep_inv_freq.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    def timestep_embedding(self, t, time_features=None):
        if time_features is None:
            time_features = self.sinusoidal_timestep_embedding(t)
        return self.timestep_fc(time_features)  # B,dim
    def embed_prompt(self, prompt):
        prompt = prompt.unsqueeze(-1)
        prompt = self.prompt_embed(prompt)
        return prompt
    def forward(self, image, prompt, timestep, time_features=None):
        # print(f"blenderinternals start fwd: {image.shape}, {prompt.shape}, {timestep.shape}")
        # image shape: B,256,dim-> B,T,C
        # prompt shape: B,1,dim
        # timestep shape: B,1
        timestep = self.timestep_embedding(
            timestep, time_features=time_features)  # B,dim
        # timestep = self.timestep_fc(timestep) #B,dim
        timestep = timestep.unsqueeze(1)  # B,1,dim
        # prompt = prompt.unsqueeze(-1) #B,1,1
        # prompt = self.prompt_embed(prompt) #B,1,dim
        # prompt embedding assumed
        # print(f"shapes: t {timestep.shape}, i {image.shape}, p {prompt.shape}")
        image = image+timestep  # B,1,dim
        prompt = prompt+timestep  # B,256,dim
        image, prompt = self.ln1(image), self.ln1(prompt)
        image_q = self.image_q(image)  # B,256,dim
        image_k = self.image_k(image)  # B,256,dim
        image_v = self.image_v(image)  # B,256,dim
        prompt_q = self.prompt_q(prompt)  # B,1,dim
        prompt_k = self.prompt_k(prompt)  # B,1,dim
        prompt_v = self.prompt_v(prompt)  # B,1,dim
        # image_q, image_k, image_v, prompt_q, prompt_k, prompt_v = self.ln1(image_q), self.ln1(image_k), self.ln1(image_v), self.ln1(prompt_q), self.ln1(prompt_k), self.ln1(prompt_v)
        Q = torch.cat([image_q, prompt_q], dim=1)  # B,257,dim
        K = torch.cat([image_k, prompt_k], dim=1)  # B,257,dim
        V = torch.cat([image_v, prompt_v], dim=1)  # B,257,dim
        # print(f"before sdpa {Q.shape}, {K.shape}, {V.shape}")
        out = torch.cat([image, prompt], dim=1)  # initialize at b,257,dim
        out = out+F.scaled_dot_product_attention(Q, K, V)  # B,257,dim
        out = self.ln2(out)
        out = out+self.mlp(out)
        out_image = out[:, :-1, :]
        out_prompt = out[:, -1, :]
        # out_prompt = self.prompt_unembed(out_prompt)
        out_prompt = out_prompt.unsqueeze(1)
        # print(f"after all",out_image.shape, out_prompt.shape)
        return out_image, out_prompt
class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=4,
                               stride=2, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(dim // 2, dim // 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(dim // 4, dim // 8,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(inplace=True),
            # Final projection to 1 channel
            nn.Conv2d(dim // 8, 1, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )
    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        H = W = int(math.sqrt(L))
        assert H * W == L, "L must be a perfect square"
        # (B, L, D) -> (B, D, H, W)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, D, H, W)
        x = self.decoder(x)
        return x
class BlenderV2(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.encoder = Encoder(dim)
        self.blenderhead = BlenderInternals(dim)
        self.blenderinternals = nn.ModuleList(
            [BlenderInternals(dim) for _ in range(num_layers-1)])
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
        time_features = self.blenderhead.sinusoidal_timestep_embedding(
            timestep)
        res_img = image
        res_prompt = prompt
        image, prompt = self.blenderhead(
            image, prompt, timestep, time_features=time_features)
        image = image+res_img
        prompt = prompt+res_prompt
        for i in range(len(self.blenderinternals)):
            res_img = image
            res_prompt = prompt
            image, prompt = self.blenderinternals[i](
                image, prompt, timestep, time_features=time_features)
            image = image+res_img
            prompt = prompt+res_prompt
        return image
    def forward_latent(self, z, prompt, t):
        prompt = self.blenderhead.embed_prompt(prompt)
        time_features = self.blenderhead.sinusoidal_timestep_embedding(t)
        res_img = z
        res_prompt = prompt
        z, prompt = self.blenderhead(z, prompt, t, time_features=time_features)
        z = z + res_img
        prompt = prompt + res_prompt
        for block in self.blenderinternals:
            res_img = z
            res_prompt = prompt
            z, prompt = block(z, prompt, t, time_features=time_features)
            z = z + res_img
            prompt = prompt + res_prompt
        return z
# model= Blender(dim=1024).to(torch.float32)
# model=StackedBlender(dim=1024, num_layers=8).to(torch.float32)
model = BlenderV2(dim=MODEL_DIM, num_layers=MODEL_LAYERS)
if LOAD_MODEL:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("model load")
model = model.to(device)
torch.set_float32_matmul_precision(MATMUL_PRECISION)
compiled_model = torch.compile(model)
# compiled_model=model
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
            # image = model.encode(image)
            # print(f"image size {image.shape}")
            t = torch.rand((image.size(0), 1), device=device)
            x1 = image
            # print(f"t {t.shape} x_t {x_t.shape} x0 {x0.shape} x1 {x1.shape}")
            with torch.autocast(device_type=device.type, dtype=AUTOMIXED_DTYPE):
                # v_predicted = compiled_model(prompt, x_t, t)
                x1 = compiled_model.encoder(x1)
                x0 = torch.randn_like(x1)
                t_broadcast = t.unsqueeze(-1)
                # print(f"t_broadcast shape {t_broadcast.shape}")
                # print(f"x0 shape {x0.shape}")
                x_t = x0 * (1 - t_broadcast) + x1 * t_broadcast
                # print(f"x-t shape{x_t.shape}")
                v_predicted = compiled_model.forward_latent(x_t, prompt, t)
                v_target = x1 - x0
                # v_target = compiled_model.encoder(v_target)
                # v_predicted = compiled_model.decoder(v_predicted)
                # print(f"-----------")
                # print(f"shapes: v_predicted {v_predicted.shape} v_target {v_target.shape}")
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
