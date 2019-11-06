"""
Copyright (c) 2019 CRISP

config

:author: Bahareh Tolooshams
"""

import torch

from sacred import Experiment, Ingredient

config_ingredient = Ingredient("cfg")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


@config_ingredient.config
def cfg():
    hyp = {
        "experiment_name": "test_mnist",
        "dataset": "MNIST",
        "network": "CRsAE2D",
        "dictionary_dim": 3,
        "stride": 1,
        "num_conv": 32,
        "L": 50,
        "num_iters": 30,
        "batch_size": 1,
        "num_epochs": 500,
        "zero_mean_filters": False,
        "normalize": True,
        "lr": 0.1,
        "lr_decay": 0.7,
        "lr_step": 10,
        "lr_lam": 0.001,
        "noiseSTD": 20,
        "shuffle": True,
        "test_path": "../data/test_img/",
        "info_period": 2000,
        "denoising": True,
        "supervised": True,
        "crop_dim": (28, 28),
        "init_with_DCT": True,
        "sigma": 0.03,
        "lam": 0.1,
        "twosided": True,
        "loss": "MSE",
        "use_lam": False,
        "delta": 100,
        "trainable_bias": False,
        "cyclic": False,
        "device": device,
    }


@config_ingredient.named_config
def crsae_voc_denoising():
    hyp = {
        "experiment_name": "crsae_voc_denoising",
        "dataset": "VOC",
        "network": "CRsAE2D",
        "dictionary_dim": 8,
        "stride": 7,
        "num_conv": 64,
        "L": 50,
        "num_iters": 30,
        "batch_size": 1,
        "num_epochs": 500,
        "zero_mean_filters": False,
        "normalize": True,
        "lr": 0.1,
        "lr_decay": 0.7,
        "lr_step": 10,
        "noiseSTD": 20,
        "shuffle": True,
        "test_path": "../data/test_img/",
        "info_period": 2000,
        "denoising": True,
        "supervised": True,
        "crop_dim": (64, 64),
        "init_with_DCT": True,
        "sigma": 0.03,
        "twosided": True,
        "loss": "MSE",
        "use_lam": False,
    }
