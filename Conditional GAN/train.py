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
import argparse

def gradient_penalty(critic, labels, real, fake, device="cuda"):
    """
    Computes the gradient penalty for WGAN-GP to enforce the 1-Lipschitz constraint.
    Interpolates between real and fake images, calculates the gradient of the critic's
    output with respect to the interpolated images, and penalizes deviations from a
    gradient norm of 1. This replaces weight clipping in the original WGAN, providing
    a more stable and flexible way to constrain the critic's gradients during training.
    """
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = critic(interpolated_images, labels)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def main():
    parser = argparse.ArgumentParser(description="Train WGAN with original or improved formulation")
    parser.add_argument(
        '--formulation',
        type=str,
        choices=['original', 'improved'],
        default='original',
        help='WGAN formulation to use: original or improved'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    IMAGE_SIZE = 64
    NUM_CLASSES = 10
    GEN_EMBEDDING = 100
    CHANNELS_IMG = 1
    Z_DIM = 100
    FEATURES_GEN = 64
    FEATURES_DISC = 64
    IMAGE_SAVE_FREQ = 1
    CRITIC_ITERATIONS = 5
    WEIGHT_CLIP = 0.01

    # Configuration based on formulation
    if args.formulation == 'original':
        LEARNING_RATE = 5e-5
        WEIGHT_CLIP = 0.01
        use_gradient_penalty = False
        use_instance_norm = False
    else:  # improved
        LEARNING_RATE = 1e-4
        LAMBDA_GP = 10
        use_gradient_penalty = True
        use_instance_norm = True

    # Directories for saving outputs
    output_dir = f"output_{args.formulation}"
    image_dir = os.path.join(output_dir, "images")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    transform_steps = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ])

    dataset = datasets.MNIST(root="dataset/", train=True, transform=transform_steps, download=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN,  NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING, use_instance_norm).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE, use_instance_norm).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # Optimizer setup based on formulation
    if args.formulation == 'original':
        gen_opt = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
        critic_opt = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)
    else:
        gen_opt = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
        critic_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)
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
        critic.load_state_dict(checkpoint['disc_state_dict'])
        gen_opt.load_state_dict(checkpoint['gen_optim_state_dict'])
        critic_opt.load_state_dict(checkpoint['disc_optim_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        gen_losses = checkpoint['gen_losses']
        disc_losses = checkpoint['disc_losses']
        best_gen_loss = checkpoint['best_gen_loss']
        print(f"Resuming training from epoch {start_epoch}")

    gen.train()
    critic.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        num_batches = 0

        for batch_idx, (real, labels) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{args.epochs}"):
            real = real.to(device)
            current_batch_size = real.shape[0]
            labels = labels.to(device)

            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn((current_batch_size, Z_DIM, 1, 1)).to(device)
                fixed_labels = torch.tensor([i % NUM_CLASSES for i in range(32)]).to(device)
                fake = gen(noise, fixed_labels)
                critic_real = critic(real, labels).reshape(-1)
                critic_fake = critic(fake.detach(), labels).reshape(-1)
                critic.zero_grad()

                if use_gradient_penalty:
                    gp = gradient_penalty(critic, labels, real, fake, device=device)
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                else:
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

                loss_critic.backward()
                critic_opt.step()

                if not use_gradient_penalty:
                    for p in critic.parameters():
                        p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            output = critic(fake, labels).reshape(-1)
            loss_gen = -torch.mean(output)
            gen.zero_grad()
            loss_gen.backward()
            gen_opt.step()

            epoch_gen_loss += loss_gen.item()
            epoch_disc_loss += loss_critic.item()
            num_batches += 1

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] Batch 0 | "
                    f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
                )

        avg_gen_loss = epoch_gen_loss / num_batches
        avg_disc_loss = epoch_disc_loss / num_batches
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] | "
            f"Avg Loss D: {avg_disc_loss:.4f}, Avg Loss G: {avg_gen_loss:.4f}"
        )

        if (epoch + 1) % IMAGE_SAVE_FREQ == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                fake = gen(fixed_noise, labels)
                img_grid_fake = vutils.make_grid(fake, normalize=True, nrow=8)
                img_grid_real = vutils.make_grid(real[:32], normalize=True, nrow=8)
                fake_img = torchvision.transforms.ToPILImage()(img_grid_fake)
                real_img = torchvision.transforms.ToPILImage()(img_grid_real)
                fake_img.save(os.path.join(image_dir, f"fake_epoch_{epoch+1}.png"))
                real_img.save(os.path.join(image_dir, f"real_epoch_{epoch+1}.png"))

        checkpoint = {
            'epoch': epoch,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': critic.state_dict(),
            'gen_optim_state_dict': gen_opt.state_dict(),
            'disc_optim_state_dict': critic_opt.state_dict(),
            'gen_losses': gen_losses,
            'disc_losses': disc_losses,
            'best_gen_loss': best_gen_loss,
            'formulation': args.formulation,
        }
        torch.save(checkpoint, checkpoint_path)

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

    print(f"Training completed! Losses saved to {output_dir}/losses.json")

if __name__ == "__main__":
    main()