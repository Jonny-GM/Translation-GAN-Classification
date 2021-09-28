# from argparse import ArgumentParser, Namespace
import itertools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core import LightningModule
from torch.nn import SELU


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            return [nn.Linear(in_feat, out_feat), SELU()]

        self.model = nn.Sequential(
            *block(latent_dim, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Linear(256, int(output_dim))
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(output_dim), 128),
            SELU(),
            nn.Linear(128, 64),
            SELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, point):
        point_flat = point.view(point.size(0), -1)
        return self.model(point_flat)


class TTGAN(LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        output_dim: int = 10,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        translation_loss_weight: float = 0.1,
        cyclic_loss_weight: float = 0,
        identity_loss_weight: float = 0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.gen_M2m = Generator(
            latent_dim=self.hparams.latent_dim,
            output_dim=self.hparams.output_dim,
        )
        self.disc_min = Discriminator(output_dim=self.hparams.output_dim)

        if self.hparams.cyclic_loss_weight:
            self.gen_m2M = Generator(
                latent_dim=self.hparams.latent_dim,
                output_dim=self.hparams.output_dim,
            )
            self.disc_maj = Discriminator(output_dim=self.hparams.output_dim)

        # if "x_maj" in kwargs and kwargs["x_maj"] is not None:
        #     self.register_buffer("x_maj", kwargs["x_maj"])

    def forward(self, z):
        return self.gen_M2m(z)

    @classmethod
    def adversarial_loss(cls, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    @classmethod
    def discriminator_loss(cls, generator, discriminator, z, points):
        valid = torch.ones(points.size(0), 1).type_as(points)
        real_loss = cls.adversarial_loss(discriminator(points), valid)
        fake = torch.zeros(z.size(0), 1).type_as(z)
        fake_loss = cls.adversarial_loss(
            discriminator(generator(z).detach()), fake
        )
        return (real_loss + fake_loss) / 2

    def training_step(self, batch, batch_idx, optimizer_idx):
        points_min = batch["min"][0]
        if (
            self.hparams.cyclic_loss_weight
            or self.hparams.translation_loss_weight
        ):
            points_maj = batch["maj"][0]

        if not self.hparams.cyclic_loss_weight:
            if not self.hparams.translation_loss_weight:
                # random noise like a regular GAN
                z_min = torch.randn(
                    points_min.shape[0], self.hparams.latent_dim
                ).to(self.device)
            else:
                z_min = points_maj
            if self.hparams.identity_loss_weight:
                z_maj = points_min
        if self.hparams.cyclic_loss_weight:
            z_min = points_maj
            z_maj = points_min

        # train generator
        if optimizer_idx == 0:

            # generate points
            generated_min = self(z_min)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(z_min.size(0), 1).type_as(z_min)

            g_loss = self.adversarial_loss(self.disc_min(generated_min), valid)
            # adversarial loss is binary cross-entropy
            if self.hparams.translation_loss_weight:
                g_loss += self.hparams.translation_loss_weight * nn.L1Loss()(
                    z_min, generated_min
                )

            if self.hparams.cyclic_loss_weight:
                generated_maj = self.gen_m2M(z_maj)
                valid = torch.ones(z_maj.size(0), 1).type_as(z_maj)
                g_loss += self.adversarial_loss(
                    self.disc_maj(generated_maj), valid
                )
                cycle_loss = F.l1_loss(
                    self.gen_m2M(generated_min), z_min
                ) + F.l1_loss(self(generated_maj), z_maj)
                g_loss += self.hparams.cyclic_loss_weight * cycle_loss

            if self.hparams.identity_loss_weight:
                id_loss = F.l1_loss(self(z_maj), z_maj)
                if self.hparams.cyclic_loss_weight:
                    id_loss += F.l1_loss(self.gen_m2M(z_min), z_min)
                g_loss += self.hparams.identity_loss_weight * id_loss

            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

        # train discriminator
        if optimizer_idx == 1:
            d_loss = self.discriminator_loss(
                self.gen_M2m, self.disc_min, z_min, points_min
            )

            if self.hparams.cyclic_loss_weight:
                d_loss += self.discriminator_loss(
                    self.gen_m2M, self.disc_maj, z_maj, points_maj
                )

            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        if self.hparams.cyclic_loss_weight:
            opt_g = torch.optim.Adam(
                itertools.chain(
                    self.gen_M2m.parameters(), self.gen_m2M.parameters()
                ),
                lr=lr,
                betas=(b1, b2),
            )
            opt_d = torch.optim.Adam(
                itertools.chain(
                    self.disc_min.parameters(), self.disc_maj.parameters()
                ),
                lr=lr,
                betas=(b1, b2),
            )
        else:
            opt_g = torch.optim.Adam(
                self.gen_M2m.parameters(), lr=lr, betas=(b1, b2)
            )
            opt_d = torch.optim.Adam(
                self.disc_min.parameters(), lr=lr, betas=(b1, b2)
            )
        return [opt_g, opt_d], []
