"""
Copyright (c) 2019 CRISP

train

:author: Bahareh Tolooshams
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
from sparselandtools.dictionaries import DCTDictionary
import os
from datetime import datetime
from sacred import Experiment

import sys

sys.path.append("src/")

import model, generator, trainer, utils, conf

from conf import config_ingredient

import warnings

warnings.filterwarnings("ignore")


ex = Experiment("train", ingredients=[config_ingredient])


@ex.automain
def run(cfg):

    hyp = cfg["hyp"]

    print(hyp)

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    PATH = "../results/{}/{}".format(hyp["experiment_name"], random_date)
    os.makedirs(PATH)

    filename = os.path.join(PATH, "hyp.pickle")
    with open(filename, "wb") as file:
        pickle.dump(hyp, file)

    print("load data.")
    if hyp["dataset"] == "VOC":
        train_loader, _ = generator.get_VOC_loaders_detection(
            hyp["batch_size"], crop_dim=hyp["crop_dim"], shuffle=hyp["shuffle"]
        )
        test_loader = generator.get_lena_loader(
            hyp["batch_size"], hyp["test_path"], shuffle=False
        )
    elif hyp["dataset"] == "MNIST":
        train_loader, test_loader = generator.get_MNIST_loaders(
            hyp["batch_size"], shuffle=hyp["shuffle"]
        )
    else:
        print("dataset is not implemented.")

    if hyp["init_with_DCT"]:
        dct_dictionary = DCTDictionary(
            hyp["dictionary_dim"], np.int(np.sqrt(hyp["num_conv"]))
        )
        H_init = dct_dictionary.matrix.reshape(
            hyp["dictionary_dim"], hyp["dictionary_dim"], hyp["num_conv"]
        ).T
        H_init = np.expand_dims(H_init, axis=1)
        H_init = torch.from_numpy(H_init).float().to(hyp["device"])
    else:
        H_init = None

    print("create model.")
    if hyp["network"] == "CRsAE1D":
        net = model.CRsAE1D(hyp, H_init)
    elif hyp["network"] == "CRsAE2D":
        net = model.CRsAE2D(hyp, H_init)
    elif hyp["network"] == "CRsAE2DTrainableBias":
        net = model.CRsAE2DTrainableBias(hyp, H_init)
    else:
        print("model does not exist!")

    torch.save(net, os.path.join(PATH, "model_init.pt"))

    if hyp["trainable_bias"]:
        criterion_ae = torch.nn.MSELoss()
        criterion_lam = utils.LambdaLoss2D()

        param_ae = []
        param_lam = []
        ctr = 0
        for param in net.parameters():
            if ctr == 2:
                param_lam.append(param)
            else:
                param_ae.append(param)

            ctr += 1

        optimizer_ae = optim.Adam(param_ae, lr=hyp["lr"], eps=1e-3)
        optimizer_lam = optim.Adam(param_lam, lr=hyp["lr_lam"], eps=1e-3)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer_ae, step_size=hyp["lr_step"], gamma=hyp["lr_decay"]
        )
    else:
        if hyp["loss"] == "MSE":
            criterion = torch.nn.MSELoss()
        elif hyp["loss"] == "L1":
            criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(net.parameters(), lr=hyp["lr"], eps=1e-3)

        if hyp["cyclic"]:
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=hyp["base_lr"],
                max_lr=hyp["max_lr"],
                step_size_up=hyp["step_size"],
                cycle_momentum=False,
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=hyp["lr_step"], gamma=hyp["lr_decay"]
            )

    print("train auto-encoder.")
    if hyp["trainable_bias"]:
        net = trainer.train_ae_withtrainablebias(
            net,
            train_loader,
            hyp,
            criterion_ae,
            criterion_lam,
            optimizer_ae,
            optimizer_lam,
            scheduler,
            PATH,
            test_loader,
            0,
            hyp["num_epochs"],
        )
    else:
        net = trainer.train_ae(
            net,
            train_loader,
            hyp,
            criterion,
            optimizer,
            scheduler,
            PATH,
            test_loader,
            0,
            hyp["num_epochs"],
        )

    print("training finished!")
