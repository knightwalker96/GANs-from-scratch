import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import torchvision.utils as vutils
from model import Discriminator, Generator, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 50
FEATURES_GEN = 64
FEATURES_DISC = 64
IMAGE_SAVE_FREQ = 1  

output_dir = "output"
image_dir = os.path.join(output_dir, "images")
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
])

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

gen_opt = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
disc_opt = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)

# Lists to track losses
gen_losses = []
disc_losses = []

checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")

# Resume training if checkpoint exists
start_epoch = 0
best_gen_loss = float('inf')
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    disc.load_state_dict(checkpoint['disc_state_dict'])
    gen_opt.load_state_dict(checkpoint['gen_optim_state_dict'])
    disc_opt.load_state_dict(checkpoint['disc_optim_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    gen_losses = checkpoint['gen_losses']
    disc_losses = checkpoint['disc_losses']
    best_gen_loss = checkpoint['best_gen_loss']
    print(f"Resuming training from epoch {start_epoch}")

gen.train()
disc.train()

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_gen_loss = 0
    epoch_disc_loss = 0
    num_batches = 0

    for batch_idx, (real, _) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        real = real.to(device)
        current_batch_size = real.shape[0]  # Handle last batch size
        noise = torch.randn((current_batch_size, Z_DIM, 1, 1)).to(device)

        # Train the Discriminator: max log(D(real)) + log(1 - D(G(noise)))
        fake = gen(noise)
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  
        loss_disc = (loss_disc_fake + loss_disc_real) / 2
        disc.zero_grad()
        loss_disc.backward()
        disc_opt.step()

        # Train the Generator: max log(D(G(z)))
        disc_fake = disc(fake).reshape(-1)
        loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))
        gen.zero_grad()
        loss_gen.backward()
        gen_opt.step()

        epoch_gen_loss += loss_gen.item()
        epoch_disc_loss += loss_disc.item()
        num_batches += 1

        if batch_idx == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch 0 | "
                f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

    avg_gen_loss = epoch_gen_loss / num_batches
    avg_disc_loss = epoch_disc_loss / num_batches
    gen_losses.append(avg_gen_loss)
    disc_losses.append(avg_disc_loss)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
        f"Avg Loss D: {avg_disc_loss:.4f}, Avg Loss G: {avg_gen_loss:.4f}"
    )

    if (epoch + 1) % IMAGE_SAVE_FREQ == 0 or epoch == NUM_EPOCHS - 1:
        with torch.no_grad():
            fake = gen(fixed_noise)
            # Normalize images to [0, 1] for saving
            img_grid_fake = vutils.make_grid(fake, normalize=True, nrow=8)
            img_grid_real = vutils.make_grid(real[:32], normalize=True, nrow=8)
            # Convert to PIL images and save
            fake_img = torchvision.transforms.ToPILImage()(img_grid_fake)
            real_img = torchvision.transforms.ToPILImage()(img_grid_real)
            fake_img.save(os.path.join(image_dir, f"fake_epoch_{epoch+1}.png"))
            real_img.save(os.path.join(image_dir, f"real_epoch_{epoch+1}.png"))

    checkpoint = {
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'gen_optim_state_dict': gen_opt.state_dict(),
        'disc_optim_state_dict': disc_opt.state_dict(),
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

print("Training completed. Losses saved to output/losses.json")
