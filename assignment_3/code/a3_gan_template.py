import argparse
import csv
import datetime
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Batch Norm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Batch Norm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Batch Norm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 784
        #   Output non-linearity
        self.model = nn.Sequential(
            nn.Linear(cfg.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor):
        # Generate images from z
        batch_size, _ = z.shape
        return self.model(z).view(batch_size, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct discriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor):
        # return discriminator score for img
        batch_size, n_channels, dim1, dim2 = img.shape
        return self.model(img.view(batch_size, dim1 * dim2))


def train(data_loader, discriminator, generator, optimizer_g, optimizer_d):
    criterion = nn.BCELoss().to(device=DEVICE)

    run = f'results/{datetime.datetime.now()}.GAN.{args.train_interval}'
    os.makedirs(run, exist_ok=True)
    os.makedirs(f'{run}/images', exist_ok=True)

    result_file = open(f'{run}/stats.csv', 'w+')
    result_writer = csv.DictWriter(result_file, fieldnames=['Epoch', 'Step', 'Loss D', 'Loss G', 'D(X)', 'D(G(Z))'])
    result_writer.writeheader()
    rows = []

    for epoch in range(args.n_epochs):
        for i, (real_images, _) in enumerate(data_loader):
            # get dimensions
            batch_size, _, _, _ = real_images.shape
            # move to cuda if necessary
            real_images = real_images.to(device=DEVICE, non_blocking=True)
            # Train Generator
            # ---------------
            # sample from normal multivariate
            z_batch = torch.randn(torch.Size([batch_size, args.latent_dim]), device=DEVICE)
            # generate samples and discriminate them
            fake_images = generator(z_batch)
            fake_outputs = discriminator(fake_images)
            # obtain generator loss
            generator_loss = criterion(fake_outputs, torch.ones(batch_size, 1, device=DEVICE))
            # train generator only every `k` steps
            if i % args.train_interval == 0:
                generator_loss.backward()
                optimizer_g.step()
                optimizer_g.zero_grad()

            # Train Discriminator
            # -------------------
            # discriminate real and fake samples
            real_outputs = discriminator(real_images)
            real_loss = criterion(real_outputs, torch.ones(batch_size, 1, device=DEVICE))
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, torch.zeros(batch_size, 1, device=DEVICE))
            # average discriminator losses
            discriminator_loss = 1 / 2 * (real_loss + fake_loss)
            # train discriminator at every step
            discriminator_loss.backward()
            optimizer_d.step()
            optimizer_d.zero_grad()

            # Save Images
            # -----------
            batches_done = epoch * len(data_loader) + i
            if batches_done % 1000 == 0:
                now = datetime.datetime.now()
                dx = float(real_outputs.sum().item()) / batch_size
                dgz = float(fake_outputs.sum().item()) / batch_size
                print("[{}]\t[Epoch: {}/{}]\t[Batches Done: {:5d}]\t[Loss D: {:.4f}]\t[Loss G: {:.4f}]"
                      "\t[D(X): {:.2f}]\t[D(G(Z)): {:.2f}]".format(now, epoch, args.n_epochs, batches_done,
                                                                   discriminator_loss.item(), generator_loss.item(), dx,
                                                                   dgz))
                rows.append({
                    'Epoch': epoch,
                    'Step': batches_done,
                    'Loss D': discriminator_loss.item(),
                    'Loss G': generator_loss.item(),
                    'D(X)': dx,
                    'D(G(Z))': dgz
                })

            if batches_done % args.save_interval == 0:
                save_image(fake_images[:25], f'{run}/images/{batches_done}.png', nrow=5, normalize=True)

        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator_optimizer': optimizer_g.state_dict(),
            'discriminator_optimizer': optimizer_d.state_dict(),
            'epoch': epoch + 1
        }, 'results/checkpoint.pth.tar')
    result_writer.writerows(rows)
    result_file.close()


def main():
    # load data
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Initialize models and optimizers
    generator = Generator(args).to(device=DEVICE)
    discriminator = Discriminator().to(device=DEVICE)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(data_loader, discriminator, generator, optimizer_g, optimizer_d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--train_interval', type=int, default=3,
                        help='Generator training interval.')
    args = parser.parse_args()

    main()
