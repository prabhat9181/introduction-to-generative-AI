import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------
# Config
# -----------------------
dataset_choice = "fashion"   # "mnist" or "fashion"
epochs = 15
batch_size = 128
noise_dim = 128
learning_rate = 0.0002
save_interval = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("generated_samples_v2", exist_ok=True)
os.makedirs("final_generated_images_v2", exist_ok=True)

# -----------------------
# Dataset
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if dataset_choice == "mnist":
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
elif dataset_choice == "fashion":
    dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

image_shape = (1, 28, 28)
image_dim = np.prod(image_shape)

# -----------------------
# Models
# -----------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, image_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), *image_shape)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        return self.model(img)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# -----------------------
# Utils
# -----------------------
def save_generated_images(epoch):
    generator.eval()
    noise = torch.randn(64, noise_dim, device=device)
    with torch.no_grad():
        images = generator(noise).cpu()
    images = (images + 1) / 2

    fig, axs = plt.subplots(8, 8, figsize=(8, 8))
    idx = 0
    for i in range(8):
        for j in range(8):
            axs[i, j].imshow(images[idx, 0], cmap="gray")
            axs[i, j].axis("off")
            idx += 1

    plt.savefig(f"generated_samples_v2/epoch_{epoch:02d}.png")
    plt.close()
    generator.train()

# -----------------------
# Training Loop
# -----------------------
for epoch in range(1, epochs + 1):
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)
        half = batch_size // 2

        optimizer_D.zero_grad()
        real_labels = torch.ones(half, 1, device=device)
        fake_labels = torch.zeros(half, 1, device=device)

        real_loss = criterion(discriminator(real_imgs[:half]), real_labels)

        noise = torch.randn(half, noise_dim, device=device)
        fake_imgs = generator(noise)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)

        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, noise_dim, device=device)
        valid_labels = torch.ones(batch_size, 1, device=device)

        g_loss = criterion(discriminator(generator(noise)), valid_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch}/{epochs} | D_loss: {d_loss.item():.3f} | G_loss: {g_loss.item():.3f}")

    if epoch % save_interval == 0:
        save_generated_images(epoch)

# -----------------------
# Save Final Images
# -----------------------
generator.eval()
noise = torch.randn(64, noise_dim, device=device)
with torch.no_grad():
    final_images = generator(noise).cpu()

final_images = (final_images + 1) / 2

for i in range(64):
    plt.imshow(final_images[i, 0], cmap="gray")
    plt.axis("off")
    plt.savefig(f"final_generated_images_v2/img_{i:03d}.png")
    plt.close()
