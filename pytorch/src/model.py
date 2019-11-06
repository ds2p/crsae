"""
Copyright (c) 2019 CRISP

crsae model

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np

import utils


class RELUTwosided(torch.nn.Module):
    def __init__(self, num_conv, lam=1e-3, L=100, sigma=1, device=None):
        super(RELUTwosided, self).__init__()
        self.L = L
        self.lam = torch.nn.Parameter(
            lam * torch.ones(1, num_conv, 1, 1, device=device)
        )
        self.sigma = sigma
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        lam = self.relu(self.lam) * (self.sigma ** 2)
        mask1 = (x > (lam / self.L)).type_as(x)
        mask2 = (x < -(lam / self.L)).type_as(x)
        out = mask1 * (x - (lam / self.L))
        out += mask2 * (x + (lam / self.L))
        return out


class CRsAE1D(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE1D, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]

        self.H = torch.nn.ConvTranspose1d(
            self.num_conv,
            1,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )
        self.HT = torch.nn.Conv1d(
            1,
            self.num_conv,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )

        if H is not None:
            self.H.weight.data = H.clone()

        self.relu = torch.nn.ReLU()

        self.H.weight.data = self.H.weight.data.to(self.device)
        self.HT.weight.data = self.H.weight.data

    def normalize(self):
        self.H.weight.data = F.normalize(self.H.weight.data, dim=-1)
        self.HT.weight.data = self.H.weight.data

    def zero_mean(self):
        self.H.weight.data -= torch.mean(self.H.weight.data, dim=0)
        self.HT.weight.data = self.H.weight.data

    def forward(self, x):
        num_batches = x.shape[0]

        D_in = x.shape[2]
        D_enc = D_in - self.dictionary_dim + 1

        self.lam = self.sigma * torch.sqrt(
            2 * torch.log(torch.zeros(1, device=self.device) + (self.num_conv * D_enc))
        )

        x_old = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        yk = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        x_new = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        t_old = torch.tensor(1, device=self.device).float()
        for t in range(self.T):
            H_wt = x - self.H(yk)
            x_new = yk + self.HT(H_wt) / self.L
            if self.twosided:
                x_new = self.relu(torch.abs(x_new) - self.lam / self.L) * torch.sign(
                    x_new
                )
            else:
                x_new = self.relu(x_new - self.lam / self.L)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = self.H(x_new)

        return z, x_new, self.lam


class CRsAE2D(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE2D, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.zero_mean_filters = hyp["zero_mean_filters"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.use_lam = hyp["use_lam"]
        if self.use_lam:
            self.lam = hyp["lam"]

        self.H = torch.nn.ConvTranspose2d(
            self.num_conv,
            1,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )
        self.HT = torch.nn.Conv2d(
            1,
            self.num_conv,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )

        if H is not None:
            self.H.weight.data = H.clone()

        self.relu = torch.nn.ReLU()

        self.H.weight.data = self.H.weight.data.to(self.device)
        self.HT.weight.data = self.H.weight.data

    def normalize(self):
        self.H.weight.data = F.normalize(self.H.weight.data, p="fro", dim=(-1, -2))
        self.HT.weight.data = self.H.weight.data

    def zero_mean(self):
        self.H.weight.data -= torch.mean(self.H.weight.data, dim=0)
        self.HT.weight.data = self.H.weight.data

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = self.HT(x_batched_padded).shape[2]
        D_enc2 = self.HT(x_batched_padded).shape[3]

        if not self.use_lam:
            self.lam = self.sigma * torch.sqrt(
                2
                * torch.log(
                    torch.tensor(
                        self.num_conv * D_enc1 * D_enc2, device=self.device
                    ).float()
                )
            )

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        yk = torch.zeros(num_batches, self.num_conv, D_enc1, D_enc2, device=self.device)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            x_new = yk + self.HT(x_batched_padded - self.H(yk)) / self.L

            if self.twosided:
                x_new = (x_new > (self.lam / self.L)).float() * (
                    x_new - (self.lam / self.L)
                ) + (x_new < -(self.lam / self.L)).float() * (
                    x_new + (self.lam / self.L)
                )
            else:
                x_new = (x_new > (self.lam / self.L)).float() * (
                    x_new - (self.lam / self.L)
                )

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (
            torch.masked_select(self.H(x_new), valids_batched.byte()).reshape(
                x.shape[0], self.stride ** 2, *x.shape[1:]
            )
        ).mean(dim=1, keepdim=False)

        return z, x_new, self.lam


class CRsAE2DTrainableBias(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE2DTrainableBias, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.zero_mean_filters = hyp["zero_mean_filters"]
        self.lam = hyp["lam"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.sigma = hyp["sigma"]

        self.H = torch.nn.ConvTranspose2d(
            self.num_conv,
            1,
            kernel_size=[self.dictionary_dim, self.dictionary_dim],
            bias=False,
        )
        self.HT = torch.nn.Conv2d(
            1,
            self.num_conv,
            kernel_size=[self.dictionary_dim, self.dictionary_dim],
            bias=False,
        )

        if H is not None:
            self.H.weight.data = H.clone()

        self.relu = RELUTwosided(
            self.num_conv, self.lam, self.L, self.sigma, self.device
        )

        self.H.weight.data = self.H.weight.data.to(self.device)
        self.HT.weight.data = self.H.weight.data

    def normalize(self):
        self.H.weight.data = F.normalize(self.H.weight.data, p="fro", dim=(-1, -2))
        self.HT.weight.data = self.H.weight.data

    def zero_mean(self):
        self.H.weight.data -= torch.mean(self.H.weight.data, dim=0)
        self.HT.weight.data = self.H.weight.data

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(I),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = self.HT(x_batched_padded).shape[2]
        D_enc2 = self.HT(x_batched_padded).shape[3]

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        ).type_as(x_batched_padded)
        yk = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        ).type_as(x_batched_padded)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        ).type_as(x_batched_padded)

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            x_new = yk + self.HT(x_batched_padded - self.H(yk)) / self.L

            x_new = self.relu(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (
            torch.masked_select(self.H(x_new), valids_batched.byte()).reshape(
                x.shape[0], self.stride ** 2, *x.shape[1:]
            )
        ).mean(dim=1, keepdim=False)

        return z, x_new, self.relu.lam
