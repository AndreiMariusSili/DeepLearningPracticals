import argparse
import os
from datetime import datetime

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as fu
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from datasets.bmnist import bmnist

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.gauss_mean = nn.Linear(hidden_dim, z_dim)
        self.gauss_log_var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        """
        Perform forward pass of encoder.

        Returns mean and log fo variance with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        embedding = self.embed(x)
        mean = self.gauss_mean(embedding)
        log_var = self.gauss_log_var(embedding)

        return mean, log_var


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Perform forward pass of decoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.model(z)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        self.bce = nn.BCELoss(reduction='sum')

    @staticmethod
    def rec_loss(x_hat: torch.Tensor, x: torch.Tensor):
        """
        Calculate the binary cross entropy loss
        between image and reconstruction.
        """
        return fu.binary_cross_entropy(x_hat, x, reduction='sum')

    @staticmethod
    def reg_loss(mu: torch.Tensor, log_var: torch.Tensor):
        """
        Calculate KL-Divergence loss between
        posterior and prior.
        """
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    @staticmethod
    def loss(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor):
        """
        Calculate full loss as the sum of
        reconstruction and regularization loss.
        """
        return VAE.rec_loss(x_hat, x) + VAE.reg_loss(mu, log_var)

    @staticmethod
    def reparameterize(mu, log_var):
        """
        Sample a value of the latent variable using the
        reparameterization trick.
        """
        eps = torch.randn(*mu.shape)
        return mu + log_var.exp().sqrt() * eps

    def forward(self, x):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        # unroll image grid to vector
        batch_size, _, _, _ = x.shape
        x = x.view(batch_size, 784)

        # forward pass with reparameterized samples
        mu, log_var = self.encoder(x)
        z = VAE.reparameterize(mu, log_var)
        x_hat = self.decoder(z)

        # calculate average elbo
        total_negative_elbo = VAE.loss(x_hat, x, mu, log_var)
        average_negative_elbo = total_negative_elbo / batch_size

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        # sample latent variable
        z = torch.randn([n_samples, self.z_dim])

        # obtain reconstruction
        x_hat = self.decoder(z)
        # sample according to the bernoulli distribution
        random_sample = torch.bernoulli(x_hat)
        # greedy sample
        greedy_sample = torch.round(x_hat)

        return greedy_sample, random_sample, x_hat


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """

    # track elbo statistics
    total_epoch_elbo = 0.0
    data_size = 0

    if model.training:
        for i, x in enumerate(data):
            batch_size, _, _, _ = x.shape
            data_size += batch_size
            average_negative_elbo = model.forward(x)
            average_negative_elbo.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_epoch_elbo += average_negative_elbo.item() * batch_size
    else:
        with torch.no_grad():
            for i, x in enumerate(data):
                batch_size, _, _, _ = x.shape
                data_size += batch_size
                average_negative_elbo = model.forward(x)

                total_epoch_elbo += average_negative_elbo.item() * batch_size
    average_epoch_elbo = total_epoch_elbo / data_size

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    train_data, val_data = data

    model.train()
    train_elbo = epoch_iter(model, train_data, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, val_data, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    run = str(datetime.now()) + f'.VAE.{ARGS.zdim}'
    run_folder = f'results/{run}'
    os.makedirs(run_folder)
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim, hidden_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    train_curve, val_curve = [None], [None]
    for epoch in range(ARGS.epochs):
        train_elbo, val_elbo = run_epoch(model, data, optimizer)
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch+1}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        greedy, random, x_hat = model.sample(25)
        save_image(greedy.view(-1, 1, 28, 28), f'{run_folder}/greedy_{epoch+1:02d}.png', nrow=5, normalize=True)
        save_image(random.view(-1, 1, 28, 28), f'{run_folder}/random_{epoch+1:02d}.png', nrow=5, normalize=True)
        save_image(x_hat.view(-1, 1, 28, 28), f'{run_folder}/reconstruction_{epoch+1:02d}.png', nrow=5, normalize=True)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        N = 10
        percentiles = np.linspace(0, 1, N)
        percentiles[0] += (percentiles[1] - percentiles[0]) / 10
        percentiles[-1] -= (percentiles[-1] - percentiles[-2]) / 10
        inv_cdf = stats.norm.ppf(percentiles)
        plt.figure(figsize=(2 * N, 2 * N), dpi=160)
        for i in range(N):
            for j in range(N):
                x_hat = model.decoder(torch.Tensor([inv_cdf[i], inv_cdf[j]]))
                plt.subplot(N, N, i * (N) + j + 1)
                plt.imshow(x_hat.view(1, 28, 28).squeeze().data.numpy(), cmap='gray')
                plt.axis('off')
        plt.savefig(f'results/{run}/manifold.png')

    save_elbo_plot(train_curve, val_curve, f'{run_folder}/elbo.pdf')
    torch.save(model.state_dict(), f'{run_folder}/model.pth.tar')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
