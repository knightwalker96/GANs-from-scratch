import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import json
import argparse
import torchvision.utils as vutils

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 784
batch_size = 32
num_epochs = 100
image_save_freq = 1

# Directories for saving outputs
output_dir = "output"
image_dir = os.path.join(output_dir, "images")
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Model, data, and optimizers
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
gen_optim = optim.Adam(gen.parameters(), lr=lr)
disc_optim = optim.Adam(disc.parameters(), lr=lr)
criterion = nn.BCELoss()

# Lists to track losses
gen_losses = []
disc_losses = []

# Checkpoint paths
checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")

parser = argparse.ArgumentParser(description="Train a GAN on MNIST")
parser.add_argument('--resume', default = True,  action='store_true', help="Resume training from the latest checkpoint")
args = parser.parse_args()

# Resume training if checkpoint exists
start_epoch = 0
best_gen_loss = float('inf')
if args.resume and os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    disc.load_state_dict(checkpoint['disc_state_dict'])
    gen_optim.load_state_dict(checkpoint['gen_optim_state_dict'])
    disc_optim.load_state_dict(checkpoint['disc_optim_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    gen_losses = checkpoint['gen_losses']
    disc_losses = checkpoint['disc_losses']
    best_gen_loss = checkpoint['best_gen_loss']
    print(f"Resuming training from epoch {start_epoch}")

# Training loop
for epoch in range(start_epoch, num_epochs):
    epoch_gen_loss = 0
    epoch_disc_loss = 0
    num_batches = 0

    for batch_idx, (real, _) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train the Discriminator: max log(D(real)) + log(1 - D(fake))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        disc_optim.step()

        # Train the Generator: max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        gen_optim.step()

        epoch_gen_loss += lossG.item()
        epoch_disc_loss += lossD.item()
        num_batches += 1

    avg_gen_loss = epoch_gen_loss / num_batches
    avg_disc_loss = epoch_disc_loss / num_batches
    gen_losses.append(avg_gen_loss)
    disc_losses.append(avg_disc_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Avg Loss D: {avg_disc_loss:.4f}, Avg Loss G: {avg_gen_loss:.4f}"
    )

    if (epoch + 1) % image_save_freq == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)
            # Normalize images to [0, 1] for saving
            img_grid_fake = vutils.make_grid(fake, normalize=True, nrow=8)
            img_grid_real = vutils.make_grid(data, normalize=True, nrow=8)
            # Convert to PIL images and save
            fake_img = torchvision.transforms.ToPILImage()(img_grid_fake)
            real_img = torchvision.transforms.ToPILImage()(img_grid_real)
            fake_img.save(os.path.join(image_dir, f"fake_epoch_{epoch+1}.png"))
            real_img.save(os.path.join(image_dir, f"real_epoch_{epoch+1}.png"))

    checkpoint = {
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'gen_optim_state_dict': gen_optim.state_dict(),
        'disc_optim_state_dict': disc_optim.state_dict(),
        'gen_losses': gen_losses,
        'disc_losses': disc_losses,
        'best_gen_loss': best_gen_loss,
    }
    torch.save(checkpoint, checkpoint_path)

    # Save best model based on generator loss
    if avg_gen_loss < best_gen_loss:
        best_gen_loss = avg_gen_loss
        torch.save(checkpoint, best_checkpoint_path)
        print(f"Saved best model at epoch {epoch+1} with generator loss {best_gen_loss:.4f}")

losses_dict = {
    'generator_losses': gen_losses,
    'discriminator_losses': disc_losses,
}
with open(os.path.join(output_dir, 'losses.json'), 'w') as f:
    json.dump(losses_dict, f, indent=4)

print("Training completed! Losses saved to output/losses.json")