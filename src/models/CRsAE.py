"""
Copyright (c) 2018 CRISP

classes related to CRsAE, etc.

:author: Bahareh Tolooshams
"""

import numpy as np
import time
from time import gmtime, strftime
import h5py
import copy
import random
import keras.backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ProgbarLogger
from keras.layers import Conv1D, Conv2D, Input, ZeroPadding2D
from keras.layers import Dense, Lambda, ZeroPadding1D
from keras.models import Model
from keras.constraints import max_norm

import sys

sys.path.append("..")

from src.layers.conv_tied_layers import Conv1DFlip, Conv2DFlip
from src.layers.ista_fista_layers import FISTA_1d, FISTA_2d
from src.layers.trainable_lambda_loss_function_layers import TrainableLambdaLossFunction
from src.callbacks.clr_callback import CyclicLR
from src.callbacks.lrfinder_callback import LRFinder

PATH = "../"


class adam_optimizer:
    def __init__(
        self,
        lr=0.0001,
        amsgrad=False,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        lambda_lr=0.5,
    ):
        self.name = "adam"
        self.lr = lr
        self.amsgrad = amsgrad
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.lambda_lr = lambda_lr

        self.update_optimizer()

    def update_optimizer(self):
        self.keras_optimizer = Adam(
            lr=self.lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            decay=self.decay,
            amsgrad=self.amsgrad,
        )
        self.keras_optimizer_for_lambda = Adam(
            lr=self.lambda_lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            decay=self.decay,
            amsgrad=self.amsgrad,
        )

    def set_lr(self, lr):
        self.lr = lr
        self.update_optimizer()

    def set_lambda_lr(self, lambda_lr):
        self.lambda_lr = lambda_lr
        self.update_optimizer()

    def set_beta_1(self, beta_1):
        self.beta_1 = beta_1
        self.update_optimizer()

    def set_beta_2(self, beta_2):
        self.beta_2 = beta_2
        self.update_optimizer()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        self.update_optimizer()

    def set_decay(self, decay):
        self.decay = decay
        self.update_optimizer()

    def set_amsgrad(self, amsgrad):
        self.amsgrad = amsgrad
        self.update_optimizer()

    def get_name(self):
        return self.name

    def get_lr(self):
        return self.lr

    def get_lambda_lr(self):
        return self.lambda_lr

    def get_beta_1(self):
        return self.beta_1

    def get_beta_2(self):
        return self.beta_2

    def get_epsilon(self):
        return self.epsilon

    def get_decay(self):
        return self.decay

    def get_amsgrad(self):
        return self.amsgrad

    def get_keras_optimizer(self):
        return self.keras_optimizer

    def get_keras_optimizer_for_lambda(self):
        return self.keras_optimizer_for_lambda


class sgd_optimizer:
    def __init__(
        self, lr=0.001, momentum=0.0, decay=0.0, nesterov=False, lambda_lr=0.001
    ):
        self.name = "sgd"
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov
        self.lambda_lr = lambda_lr

        self.update_optimizer()

    def update_optimizer(self):
        self.keras_optimizer = SGD(
            lr=self.lr, momentum=self.momentum, decay=self.decay, nesterov=self.nesterov
        )
        self.keras_optimizer_for_lambda = SGD(
            lr=self.lambda_lr,
            momentum=self.momentum,
            decay=self.decay,
            nesterov=self.nesterov,
        )

    def set_lr(self, lr):
        self.lr = lr
        self.update_optimizer()

    def set_lambda_lr(self, lambda_lr):
        self.lambda_lr = lambda_lr
        self.update_optimizer()

    def set_momentum(self, momentum):
        self.momentum = momentum
        self.update_optimizer()

    def set_decay(self, decay):
        self.decay = decay
        self.update_optimizer()

    def set_nesterov(self, nesterov):
        self.nesterov = nesterov
        self.update_optimizer()

    def get_name(self):
        return self.name

    def get_lr(self):
        return self.lr

    def get_lambda_lr(self):
        return self.lambda_lr

    def get_momentum(self):
        return self.momentum

    def get_decay(self):
        return self.decay

    def get_nesterov(self):
        return self.nesterov

    def get_keras_optimizer(self):
        return self.keras_optimizer

    def get_keras_optimizer_for_lambda(self):
        return self.keras_optimizer_for_lambda


class trainer:
    def __init__(
        self,
        lambda_trainable,
        num_epochs=10,
        num_val_shuffle=1,
        batch_size=32,
        verbose=1,
    ):
        # training parameters
        self.lambda_trainable = lambda_trainable
        self.num_epochs = num_epochs
        self.num_val_shuffle = num_val_shuffle
        self.batch_size = batch_size
        self.verbose = verbose

        self.val_split = 0.9
        self.unique_number = int(time.time())
        self.fit_time = 0
        self.loss = "mse"
        self.history = []
        self.H_epochs = []
        self.lambda_epochs = []
        self.noiseSTD_epochs = []
        self.close = False
        self.augment = False

        self.reset_callbacks()

        self.set_optimizer()

    def add_best_val_loss_callback(self, loss_type):
        self.loss_type = loss_type
        self.callbacks.append(
            ModelCheckpoint(
                filepath="weights_{}.hdf5".format(self.unique_number),
                monitor=self.loss_type,
                verbose=self.verbose,
                save_best_only=True,
                save_weights_only=True,
            )
        )

    def add_all_epochs_callback(self, loss_type):
        self.loss_type = loss_type
        self.callbacks.append(
            ModelCheckpoint(
                filepath="weights-improvement-%i-{epoch:01d}.hdf5" % self.unique_number,
                monitor=self.loss_type,
                verbose=0,
                save_weights_only=True,
            )
        )

    def add_earlystopping_callback(self, min_delta, patience, loss_type):
        self.earlystopping = True
        self.min_delta = min_delta
        self.patience = patience
        self.loss_type = loss_type
        self.callbacks.append(
            EarlyStopping(
                monitor=self.loss_type,
                min_delta=self.min_delta,
                patience=self.patience,
                verbose=0,
                mode="auto",
            )
        )

    def add_cyclic_lr_callback(self, base_lr, max_lr, step_size, cycle_mode, gamma=1):
        self.cycleLR = True
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.cycle_mode = cycle_mode
        self.gamma = gamma
        if self.cycle_mode == "exp_range":
            self.clr = CyclicLR(
                base_lr=self.base_lr,
                max_lr=self.max_lr,
                step_size=self.step_size,
                mode=self.cycle_mode,
                gamma=self.gamma,
            )
        self.callbacks.append(self.clr)

    def add_progressbar_callback(self):
        self.callbacks.append(ProgbarLogger())

    def reset_callbacks(self):
        self.earlystopping = False
        self.cycleLR = False
        self.callbacks = []

    def get_callbacks(self):
        return self.callbacks

    def get_verbose(self):
        return self.verbose

    def get_batch_size(self):
        return self.batch_size

    def get_num_epochs(self):
        return self.num_epochs

    def get_num_train_reset(self):
        return self.num_train_reset

    def get_cycleLR(self):
        return self.cycleLR

    def get_loss(self):
        return self.loss

    def get_callbacks(self):
        return self.callbacks

    def get_callback_loss_type(self):
        return self.loss_type

    def get_fit_time(self):
        return self.fit_time

    def get_val_split(self):
        return self.val_split

    def get_unique_number(self):
        return self.unique_number

    def get_history(self):
        return self.history

    def get_H_epochs(self):
        return self.H_epochs

    def get_lambda_epochs(self):
        return self.lambda_epochs

    def get_noiseSTD_epochs(self):
        return self.noiseSTD_epochs

    def get_close(self):
        return self.close

    def get_augment(self):
        return self.augment

    def get_lr_iterations(self):
        print(self.clr.history)
        return self.clr.history["lr"]

    # optimizer
    def set_optimizer(self, name="adam"):
        if name == "adam":
            self.optimizer = adam_optimizer()
        elif name == "SGD":
            self.optimizer = sgd_optimizer()

    def get_optimizer(self):
        return self.optimizer

    def set_loss(self, loss):
        self.loss = loss

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs

    def set_num_train_reset(self, num_train_reset):
        self.num_train_reset = num_train_reset

    def set_fit_time(self, fit_time):
        self.fit_time = fit_time

    def set_val_split(self, val_split):
        self.val_split = val_split

    def set_history(self, history):
        if self.lambda_trainable:
            val_loss = history["val_loss"]
            val_loss = [float(i) for i in val_loss]

            val_lambda_loss = history["val_lambda_loss"]
            val_lambda_loss = [float(i) for i in val_lambda_loss]

            train_loss = history["train_loss"]
            train_loss = [float(i) for i in train_loss]

            train_lambda_loss = history["train_lambda_loss"]
            train_lambda_loss = [float(i) for i in train_lambda_loss]
        else:
            val_loss = history.history["val_loss"]
            val_loss = [float(i) for i in val_loss]

            train_loss = history.history["loss"]
            train_loss = [float(i) for i in train_loss]

        if self.lambda_trainable:
            history = {
                "val_loss": val_loss,
                "val_lambda_loss": val_lambda_loss,
                "train_loss": train_loss,
                "train_lambda_loss": train_lambda_loss,
            }
        else:
            history = {"val_loss": val_loss, "train_loss": train_loss}
        self.history = history

    def set_close(self, close):
        self.close = close

    def set_augment(self, augment):
        self.augment = augment


class CRsAE_1d:
    def __init__(
        self,
        input_dim,
        num_conv,
        dictionary_dim,
        num_iterations,
        L,
        twosided,
        lambda_trainable,
        MIMO,
        alpha,
        num_channels,
        noiseSTD_trainable=False,
        lambda_EM=False,
        delta=100,
        lambda_single=False,
        noiseSTD_lr=0.01,
    ):
        """
        Initializes CRsAE 1d with model and training parameters.

        """
        # models parameters
        self.input_dim = input_dim
        self.num_conv = num_conv
        self.dictionary_dim = dictionary_dim
        self.num_iterations = num_iterations
        self.L = L
        self.twosided = twosided
        self.lambda_trainable = lambda_trainable
        self.MIMO = MIMO
        self.Ne = self.input_dim - self.dictionary_dim + 1
        self.alpha = alpha
        self.num_channels = num_channels
        self.noiseSTD_trainable = noiseSTD_trainable
        self.data_space = 1
        self.delta = delta
        self.lambda_single = lambda_single
        self.lambda_EM = lambda_EM
        self.noiseSTD_lr = noiseSTD_lr

        # initialize trainer
        self.trainer = trainer(self.lambda_trainable)

    def build_model(self, noiseSTD):
        print("build model.")
        # compute lambda from noise level
        self.noiseSTD = np.zeros((1,)) + noiseSTD
        lambda_donoho = np.sqrt(2 * np.log(self.num_conv * self.Ne)) / noiseSTD
        self.lambda_donoho = np.copy(lambda_donoho)

        if self.lambda_trainable:
            # build model
            build_graph_start_time = time.time()
            autoencoder, model_2, encoder, decoder = self.CRsAE_1d_model()
            build_graph_time = time.time() - build_graph_start_time
            print("build_graph_time:", np.round(build_graph_time, 4), "s")
            if self.lambda_single:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((1,), dtype=np.float32) + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
            else:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((self.num_conv,), dtype=np.float32)
                        + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
            if self.lambda_EM:
                if self.lambda_single:
                    model_2.get_layer("logLambdaLoss").set_weights(
                        [np.zeros((1,), dtype=np.float32) + self.lambda_donoho]
                    )
                else:
                    model_2.get_layer("logLambdaLoss").set_weights(
                        [
                            np.zeros((self.num_conv,), dtype=np.float32)
                            + self.lambda_donoho
                        ]
                    )
            else:
                if self.lambda_single:
                    model_2.get_layer("FISTA").set_weights(
                        [
                            np.zeros((1,), dtype=np.float32) + self.lambda_donoho,
                            self.noiseSTD,
                        ]
                    )
                else:
                    model_2.get_layer("FISTA").set_weights(
                        [
                            np.zeros((self.num_conv,), dtype=np.float32)
                            + self.lambda_donoho,
                            self.noiseSTD,
                        ]
                    )
        else:
            # build model
            build_graph_start_time = time.time()
            autoencoder, encoder, decoder = self.CRsAE_1d_model()
            build_graph_time = time.time() - build_graph_start_time
            print("build_graph_time:", np.round(build_graph_time, 4), "s")
            # this is for training purposes with alpha
            autoencoder_for_train, temp, temp = self.CRsAE_1d_model()
            if self.lambda_single:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((1,), dtype=np.float32) + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
                autoencoder_for_train.get_layer("FISTA").set_weights(
                    [
                        (np.zeros((1,), dtype=np.float32) + self.lambda_donoho)
                        * self.alpha,
                        self.noiseSTD,
                    ]
                )
            else:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((self.num_conv,), dtype=np.float32)
                        + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
                autoencoder_for_train.get_layer("FISTA").set_weights(
                    [
                        (
                            np.zeros((self.num_conv,), dtype=np.float32)
                            + self.lambda_donoho
                        )
                        * self.alpha,
                        self.noiseSTD,
                    ]
                )

        self.encoder = encoder
        self.autoencoder = autoencoder
        self.decoder = decoder

        if self.lambda_trainable:
            self.model_2 = model_2
        else:
            self.autoencoder_for_train = autoencoder_for_train

        # initialize H
        self.initialize_H()

    def CRsAE_1d_model(self):
        """
        Create DAG for constraint reccurent sparse auto-encoder (CRsAE).
        :return: (autoencoder, encoder, decoder)
        """
        y = Input(shape=(self.input_dim, 1), name="y")

        # Zero-pad layer
        padding_layer = ZeroPadding1D(padding=(self.dictionary_dim - 1), name="zeropad")

        if self.lambda_trainable:
            ###########
            # model 1
            ###########
            # Dictionary for model 1 (trainable)
            HT_1 = Conv1D(
                filters=self.num_conv,
                kernel_size=self.dictionary_dim,
                padding="valid",
                use_bias=False,
                activation=None,
                trainable=True,
                input_shape=(self.input_dim, 1),
                name="HT",
                # kernel_constraint=max_norm(max_value=1, axis=0),
            )
            # HTy for model 1
            HTy_1 = HT_1(y)
            # initialize z0 to be 0 vector for model 1
            z0_model1 = Lambda(lambda x: x * 0, name="initialize_z")(HTy_1)
            # Have to define the transpose layer after creating a flowgraph with HT for model 1
            H_model1 = Conv1DFlip(HT_1, name="H")
            # perform FISTA for num_iterations
            # FISTA layer for the model trianing H
            FISTA_layer_model_1 = FISTA_1d(
                HT_1,
                y,
                self.L,
                False,
                self.twosided,
                self.num_iterations,
                self.lambda_single,
                False,
                name="FISTA",
            )
            # z_T output of soft-thresholding
            zt_model1 = FISTA_layer_model_1(z0_model1)

            # zeropad zt for model 1
            zt_padded_1 = padding_layer(zt_model1)
            # reconstruct y for model 1
            y_hat_model1 = H_model1(zt_padded_1)

            # build encoder for model 1
            encoder_model1 = Model(y, zt_model1)
            # build autoencoder for model 1
            autoencoder_model1 = Model(y, y_hat_model1)

            ###########
            # model 2
            ###########
            if self.lambda_EM:
                z = Input(shape=(self.num_conv,), name="l1_norm_code")
                logLoss_layer = TrainableLambdaLossFunction(
                    self.Ne,
                    self.num_conv,
                    self.lambda_donoho,
                    self.delta,
                    self.lambda_single,
                    name="logLambdaLoss",
                )
                log_loss = logLoss_layer(z)
                model_2 = Model(z, log_loss)
            else:
                HT_2 = Conv1D(
                    filters=self.num_conv,
                    kernel_size=self.dictionary_dim,
                    padding="valid",
                    use_bias=False,
                    activation=None,
                    trainable=False,
                    input_shape=(self.input_dim, 1),
                    name="HT",
                    # kernel_constraint=max_norm(max_value=1, axis=0),
                )
                # HTy for model 2
                HTy_2 = HT_2(y)
                # initialize z0 to be 0 vector for model 2
                z0_model2 = Lambda(lambda x: x * 0, name="initialize_z")(HTy_2)
                # Have to define the transpose layer after creating a flowgraph with HT for model 2
                H_model2 = Conv1DFlip(HT_2, name="H")
                # FISTA layer for the model trianing lambda
                FISTA_layer_model_2 = FISTA_1d(
                    HT_2,
                    y,
                    self.L,
                    True,
                    self.twosided,
                    self.num_iterations,
                    self.lambda_single,
                    self.lambda_EM,
                    name="FISTA",
                )
                # z_T output of soft-thresholding
                zt_model2, lambda_term_model2 = FISTA_layer_model_2(z0_model2)
                # for model 2
                lambda_placeholder = Lambda(
                    lambda x: x[:, 0, :] * 0 + lambda_term_model2, name="lambda"
                )(z0_model2)
                lambda_zt = Lambda(lambda x: x * lambda_term_model2, name="l1_norm")(
                    zt_model2
                )

                # build model for model 2
                # zeropad zt for model 2
                zt_padded_2 = padding_layer(zt_model2)
                # reconstruct y for model 2
                y_hat_model2 = H_model1(zt_padded_2)
                model_2 = Model(y, [lambda_zt, lambda_placeholder, lambda_placeholder])

            # for decoding
            input_code = Input(shape=(self.Ne, self.num_conv), name="input_code")
            input_code_padded = padding_layer(input_code)
            decoded = H_model1(input_code_padded)

            decoder = Model(input_code, decoded)

            return autoencoder_model1, model_2, encoder_model1, decoder
        else:
            HT = Conv1D(
                filters=self.num_conv,
                kernel_size=self.dictionary_dim,
                padding="valid",
                use_bias=False,
                activation=None,
                trainable=True,
                input_shape=(self.input_dim, 1),
                name="HT",
                kernel_constraint=max_norm(max_value=1, axis=0),
            )
            # HTy
            HTy = HT(y)
            # initialize z0 to be 0 vector
            z0 = Lambda(lambda x: x * 0, name="initialize_z")(HTy)

            # Have to define the transpose layer after creating a flowgraph with HT
            H = Conv1DFlip(HT, name="H")
            # perform FISTA for num_iterations
            # FISTA layer
            FISTA_layer = FISTA_1d(
                HT,
                y,
                self.L,
                False,
                self.twosided,
                self.num_iterations,
                self.lambda_single,
                False,
                name="FISTA",
            )
            zt = FISTA_layer(z0)
            # zeropad zt
            zt_padded = padding_layer(zt)
            # reconstruct y
            y_hat = H(zt_padded)

            # build encoder and autoencoder
            encoder = Model(y, zt)
            autoencoder = Model(y, y_hat)

            # for decoding
            input_code = Input(
                shape=(self.input_dim - self.dictionary_dim + 1, self.num_conv),
                name="input_code",
            )
            input_code_padded = padding_layer(input_code)
            decoded = H(input_code_padded)

            decoder = Model(input_code, decoded)

            return autoencoder, encoder, decoder

    def initialize_H(self):
        self.H = np.random.randn(self.dictionary_dim, 1, self.num_conv)
        self.H /= np.linalg.norm(self.H, axis=0)
        # set H for autoencoder and encoder
        self.autoencoder.get_layer("HT").set_weights([self.H])
        # set H for model2 if lambda is lambda_trainable
        if self.lambda_trainable:
            if not self.lambda_EM:
                self.model_2.get_layer("HT").set_weights([self.H])
        else:
            self.autoencoder_for_train.get_layer("HT").set_weights([self.H])

    def set_H(self, H, H_noisestd=0):
        if np.sum(H_noisestd):
            H_noisy = np.copy(H)
            for n in range(self.num_conv):
                H_noisy[:, :, n] += H_noisestd[n] * np.random.randn(
                    self.dictionary_dim, 1
                )
                self.H = H_noisy
        else:
            self.H = H
        self.H /= np.linalg.norm(self.H, axis=0)
        # set HT in autoencoder
        self.autoencoder.get_layer("HT").set_weights([self.H])
        if self.lambda_trainable:
            if not self.lambda_EM:
                self.model_2.get_layer("HT").set_weights([self.H])
        else:
            self.autoencoder_for_train.get_layer("HT").set_weights([self.H])

    def set_lambda(self, lambda_value):
        noiseSTD_estimate = self.autoencoder.get_layer("FISTA").get_weights()[1]
        self.autoencoder.get_layer("FISTA").set_weights(
            [lambda_value, noiseSTD_estimate]
        )
        if self.lambda_trainable:
            if self.lambda_EM:
                self.model_2.get_layer("logLambdaLoss").set_weights([lambda_value])
            else:
                noiseSTD_estimate = self.model_2.get_layer("FISTA").get_weights()[1]
                self.model_2.get_layer("FISTA").set_weights(
                    [lambda_value, noiseSTD_estimate]
                )
        else:
            noiseSTD_estimate = self.autoencoder_for_train.get_layer(
                "FISTA"
            ).get_weights()[1]
            self.autoencoder_for_train.get_layer("FISTA").set_weights(
                [lambda_value, noiseSTD_estimate]
            )

    def set_noiseSTD(self, noiseSTD):
        lambda_value = self.autoencoder.get_layer("FISTA").get_weights()[0]
        self.autoencoder.get_layer("FISTA").set_weights([lambda_value, noiseSTD])
        if self.lambda_trainable:
            if not self.lambda_EM:
                lambda_value = self.model_2.get_layer("FISTA").get_weights()[0]
                self.model_2.get_layer("FISTA").set_weights([lambda_value, noiseSTD])
        else:
            lambda_value = self.autoencoder_for_train.get_layer("FISTA").get_weights()[
                0
            ]
            self.autoencoder_for_train.get_layer("FISTA").set_weights(
                [lambda_value, noiseSTD]
            )

    def get_H(self):
        return self.autoencoder.get_layer("HT").get_weights()[0]

    def get_H_estimate(self):
        return self.autoencoder.get_layer("HT").get_weights()[0]

    def get_lambda(self):
        lambda_value = self.autoencoder.get_layer("FISTA").get_weights()[0]
        return lambda_value

    def get_lambda_estimate(self):
        if self.lambda_EM:
            lambda_value = self.model_2.get_layer("logLambdaLoss").get_weights()[0]
        else:
            lambda_value = self.model_2.get_layer("FISTA").get_weights()[0]
        return lambda_value

    def get_noiseSTD(self):
        noiseSTD = self.autoencoder.get_layer("FISTA").get_weights()[1]
        return noiseSTD

    def get_input_dim(self):
        return self.input_dim

    def get_num_conv(self):
        return self.num_conv

    def get_dictionary_dim(self):
        return self.dictionary_dim

    def get_num_iterations(self):
        return self.num_iterations

    def get_L(self):
        return self.L

    def get_twosided(self):
        return self.twosided

    def get_MIMO(self):
        return self.MIMO

    def get_alpha(self):
        return self.alpha

    def get_num_channels(self):
        return self.num_channels

    def get_data_space(self):
        return self.data_space

    def update_H_after_training(self):
        if self.lambda_trainable:
            # load parameters from the best val_loss
            self.autoencoder.load_weights(
                "weights_{}.hdf5".format(self.trainer.get_unique_number())
            )
            self.set_H(self.autoencoder.get_layer("HT").get_weights()[0])
        else:
            # load parameters from the best val_loss
            self.autoencoder_for_train.load_weights(
                "weights_{}.hdf5".format(self.trainer.get_unique_number())
            )
            self.set_H(self.autoencoder_for_train.get_layer("HT").get_weights()[0])

    def update_H_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.H_epochs = []
        for epoch in range(num_epochs):
            if self.lambda_trainable:
                self.autoencoder.load_weights(
                    "weights-improvement-{}-{}.hdf5".format(
                        self.trainer.get_unique_number(), epoch + 1
                    )
                )
                self.trainer.H_epochs.append(
                    self.autoencoder.get_layer("HT").get_weights()[0]
                )
            else:
                self.autoencoder_for_train.load_weights(
                    "weights-improvement-{}-{}.hdf5".format(
                        self.trainer.get_unique_number(), epoch + 1
                    )
                )
                self.trainer.H_epochs.append(
                    self.autoencoder_for_train.get_layer("HT").get_weights()[0]
                )

    def update_H(self):
        H_estimate = self.get_H_estimate()
        self.H = H_estimate
        self.model_2.get_layer("HT").set_weights([self.H])

    def update_lambda(self):
        lambda_value = self.get_lambda_estimate()
        noiseSTD_estimate = self.autoencoder.get_layer("FISTA").get_weights()[1]
        self.autoencoder.get_layer("FISTA").set_weights(
            [lambda_value, noiseSTD_estimate]
        )

    def update_lambda_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.lambda_epochs = []
        for epoch in range(num_epochs):
            self.autoencoder.load_weights(
                "weights-improvement-{}-{}.hdf5".format(
                    self.trainer.get_unique_number(), epoch + 1
                )
            )
            self.trainer.lambda_epochs.append(
                self.autoencoder.get_layer("FISTA").get_weights()[0]
            )

    def update_noiseSTD_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.noiseSTD_epochs = []
        for epoch in range(num_epochs):
            self.autoencoder.load_weights(
                "weights-improvement-{}-{}.hdf5".format(
                    self.trainer.get_unique_number(), epoch + 1
                )
            )
            self.trainer.noiseSTD_epochs.append(
                self.autoencoder.get_layer("FISTA").get_weights()[1][0]
            )

    def encode(self, y):
        return self.encoder.predict(y)

    def decode(self, z):
        return self.decoder.predict(z)

    def separate(self, y):
        # get z
        z = self.encode(y)

        y_hat_separate = np.zeros((y.shape[0], self.input_dim, self.num_conv))
        for n in range(self.num_conv):
            temp = np.copy(np.zeros((z.shape[0], self.Ne, self.num_conv)))
            temp[:, :, n] = np.copy(z[:, :, n])
            decoded = copy.deepcopy(self.decode(temp))
            y_hat_separate[:, :, n] = np.squeeze(np.copy(decoded), axis=2)
        return y_hat_separate

    def denoise(self, y):
        return self.autoencoder.predict(y)

    def compile(self):
        def lambda_loss_function(y_true, y_pred):
            return K.sum(y_pred, axis=-1)

        def log_lambda_loss_function(y_true, y_pred):
            return K.sum(K.log(y_pred), axis=-1)

        def lambda_gamma_prior_loss_function(y_true, y_pred):
            delta = self.delta
            r = delta * self.lambda_donoho
            return K.sum((r - 1) * K.log(y_pred) - delta * y_pred, axis=-1)

        loss = self.trainer.get_loss()

        if self.lambda_trainable:
            self.autoencoder.compile(
                optimizer=self.trainer.optimizer.get_keras_optimizer(),
                loss=loss,
                loss_weights=[1 / 2],
            )
            if self.lambda_EM:
                self.model_2.compile(
                    optimizer=self.trainer.optimizer.get_keras_optimizer_for_lambda(),
                    loss=lambda_loss_function,
                )
            else:
                loss_model2 = [
                    "mae",
                    log_lambda_loss_function,
                    lambda_gamma_prior_loss_function,
                ]
                l1_norm_lambda_loss_weight = 1
                log_lambda_loss_weight = -self.Ne
                lambda_gamma_prior_loss_weight = -1
                self.model_2.compile(
                    optimizer=self.trainer.optimizer.get_keras_optimizer_for_lambda(),
                    loss=loss_model2,
                    loss_weights=[
                        l1_norm_lambda_loss_weight,
                        log_lambda_loss_weight,
                        lambda_gamma_prior_loss_weight,
                    ],
                )

        else:
            self.autoencoder_for_train.compile(
                optimizer=self.trainer.optimizer.get_keras_optimizer(), loss=loss
            )

    def train_on_batch(self, training_batch_i):
        # single gradient update for H
        x, y, sample_weights = self.autoencoder._standardize_user_data(
            training_batch_i, training_batch_i, sample_weight=None, class_weight=None
        )
        if self.autoencoder._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [1.0]
        else:
            ins = x + y + sample_weights
        output_1 = self.autoencoder.train_function(ins)

        if self.lambda_EM:
            training_batch_i_code = self.encode(training_batch_i)
            l1_norm_training_batch_i_code = np.mean(
                np.sum(np.abs(training_batch_i_code), axis=1), axis=0
            )
            # single gradient update for lambda
            x, y, sample_weights = self.model_2._standardize_user_data(
                np.zeros((1, self.num_conv)) + l1_norm_training_batch_i_code,
                [np.zeros((1, 1))],
                sample_weight=None,
                class_weight=None,
            )
        else:
            # single gradient update for lambda
            x, y, sample_weights = self.model_2._standardize_user_data(
                training_batch_i,
                [
                    np.zeros((training_batch_i.shape[0], self.Ne, self.num_conv)),
                    np.zeros((training_batch_i.shape[0], self.num_conv)),
                    np.zeros((training_batch_i.shape[0], self.num_conv)),
                ],
                sample_weight=None,
                class_weight=None,
            )

        if self.model_2._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [1.0]
        else:
            ins = x + y + sample_weights
        output_2 = self.model_2.train_function(ins)

    def fit(self, partial_y_train, y_val, lr_finder=[], num_epochs=4):
        if self.lambda_trainable:
            num_batches = np.ceil(
                partial_y_train.shape[0] / self.trainer.get_batch_size()
            )
            val_loss = []
            train_loss = []
            val_lambda_loss = []
            train_lambda_loss = []
            print(
                "Train on %i samples, validate on %i samples"
                % (partial_y_train.shape[0], y_val.shape[0])
            )

            # this is something to do for compile (before start training)
            self.autoencoder._make_train_function()
            self.model_2._make_train_function()

            for epoch in range(self.trainer.get_num_epochs()):
                print("noiseSTD:", self.get_noiseSTD()[0])
                print("lambda:", self.get_lambda())
                batches = np.linspace(0, num_batches - 1, num_batches)
                random.shuffle(batches)
                for batch in batches:
                    batch_begin_index = np.int(batch * self.trainer.get_batch_size())
                    batch_end_index = np.int(
                        (batch + 1) * self.trainer.get_batch_size()
                    )
                    training_batch_i = partial_y_train[
                        batch_begin_index:batch_end_index, :, :
                    ]

                    if self.noiseSTD_trainable:
                        # get noiseSTD (l+1) estimate from H (l), lambda (l), noiseSTD (l) by gradient descent
                        # get reconstruction loss on batch
                        training_batch_i_hat = self.denoise(training_batch_i)

                        noiseSTD_estimate_past = self.get_noiseSTD()[0]

                        beta = 1 - self.noiseSTD_lr
                        noise_estimate_from_batch = np.sqrt(
                            np.mean(
                                np.mean(
                                    np.square(
                                        np.squeeze(
                                            training_batch_i - training_batch_i_hat
                                        )
                                    ),
                                    axis=1,
                                )
                            )
                        )
                        noiseSTD_estimate = np.zeros((1,)) + (
                            beta * noiseSTD_estimate_past
                            + (1 - beta) * noise_estimate_from_batch
                        )

                    # train on single batch
                    self.train_on_batch(training_batch_i)

                    # # update lambda on autoencoder
                    self.update_lambda()
                    # # # update H on model 2
                    if not self.lambda_EM:
                        self.update_H()

                    test = self.get_H()
                    self.set_H(test)

                    # update noiseSTD
                    if self.noiseSTD_trainable:
                        self.set_noiseSTD(noiseSTD_estimate)

                # keep track of weights for all epochs
                self.autoencoder.save_weights(
                    "weights-improvement-%i-%i.hdf5"
                    % (self.trainer.get_unique_number(), epoch + 1),
                    overwrite=True,
                )

                # test on validation set
                y_val_hat = self.denoise(y_val)
                val_loss_i = np.mean(
                    np.mean(np.square(np.squeeze(y_val - y_val_hat)), axis=1), axis=0
                )
                # get training error
                partial_y_train_hat = self.denoise(partial_y_train)
                train_loss_i = np.mean(
                    np.mean(
                        np.square(np.squeeze(partial_y_train - partial_y_train_hat)),
                        axis=1,
                    ),
                    axis=0,
                )

                # lambda loss on validation set
                partial_x_train_hat = self.encode(partial_y_train)
                l1_norm_partial_x_train_hat = np.mean(
                    np.sum(np.abs(partial_x_train_hat), axis=1), axis=0
                )
                if self.lambda_EM:
                    train_lambda_loss_i = self.model_2.predict(
                        np.zeros((1, self.num_conv)) + l1_norm_partial_x_train_hat
                    )
                else:
                    train_lambda_loss_i = 0

                # lambda loss on training set
                x_val_hat = self.encode(y_val)
                l1_norm_x_val_hat = np.mean(np.sum(np.abs(x_val_hat), axis=1), axis=0)
                if self.lambda_EM:
                    val_lambda_loss_i = self.model_2.predict(
                        np.zeros((1, self.num_conv)) + l1_norm_x_val_hat
                    )
                else:
                    val_lambda_loss_i = 0

                val_loss.append(val_loss_i)
                val_lambda_loss.append(train_lambda_loss_i)
                train_loss.append(train_loss_i)
                train_lambda_loss.append(train_lambda_loss_i)

                print(
                    "Epoch %d/%d" % (epoch + 1, self.trainer.get_num_epochs()),
                    "loss:",
                    np.round(train_loss_i, 8),
                    "lambda_loss:",
                    np.round(train_lambda_loss_i, 3),
                    "val_loss:",
                    np.round(val_loss_i, 8),
                    "val_lambda_loss:",
                    np.round(val_lambda_loss_i, 3),
                )

                if epoch == 0:
                    self.autoencoder.save_weights(
                        "weights_{}.hdf5".format(self.trainer.get_unique_number()),
                        overwrite=True,
                    )
                    print(
                        "val_loss improved from inf to %.8f saving model to %s"
                        % (
                            val_loss_i,
                            "weights_{}.hdf5".format(self.trainer.get_unique_number()),
                        )
                    )
                else:
                    if val_loss_i < min_val_loss:
                        self.autoencoder.save_weights(
                            "weights_{}.hdf5".format(self.trainer.get_unique_number()),
                            overwrite=True,
                        )
                        print(
                            "val_loss improved from %.8f to %.8f saving model to %s"
                            % (
                                val_loss[-2],
                                val_loss_i,
                                "weights_{}.hdf5".format(
                                    self.trainer.get_unique_number()
                                ),
                            )
                        )
                    else:
                        print("val_loss NOT improved from %.8f" % (min_val_loss))

                min_val_loss = min(val_loss)

            history = {
                "val_loss": val_loss,
                "val_lambda_loss": val_lambda_loss,
                "train_loss": train_loss,
                "train_lambda_loss": train_lambda_loss,
            }
        else:
            if lr_finder:
                history = self.autoencoder_for_train.fit(
                    partial_y_train,
                    partial_y_train,
                    epochs=num_epochs,
                    batch_size=self.trainer.get_batch_size(),
                    validation_data=(y_val, y_val),
                    verbose=self.trainer.get_verbose(),
                    shuffle=True,
                    callbacks=[lr_finder],
                )
            else:
                history = self.autoencoder_for_train.fit(
                    partial_y_train,
                    partial_y_train,
                    epochs=np.int(
                        self.trainer.get_num_epochs()
                        / self.trainer.get_num_train_reset()
                    ),
                    batch_size=self.trainer.get_batch_size(),
                    validation_data=(y_val, y_val),
                    verbose=self.trainer.get_verbose(),
                    shuffle=True,
                    callbacks=self.trainer.get_callbacks(),
                )
        return history

    def find_lr(self, y_train, folder_name, num_epochs=4, min_lr=1e-7, max_lr=1e-1):
        # This implementation is ONLY for LR for autoencoder (H training) not lambda model (model 2)
        print("find lr.")

        num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)
        y_val = y_train[indices[partial_train_num:], :, :]

        partial_y_train_original = y_train[indices[:partial_train_num], :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        # lr callback
        epoch_size = int(0.9 * num_train)
        lr_finder = LRFinder(
            min_lr=min_lr,
            max_lr=max_lr,
            steps_per_epoch=np.ceil(epoch_size / self.trainer.get_batch_size()),
            epochs=num_epochs,
        )

        # compile model with loss (weighted loss)
        self.compile()
        # fit (train)
        history = self.fit(partial_y_train, y_val, lr_finder, num_epochs)

        # save lr results
        hf = h5py.File(
            "../experiments/{}/results/results_lr.h5".format(folder_name), "w"
        )
        hf.create_dataset("iterations", data=lr_finder.get_iterations())
        hf.create_dataset("lr", data=lr_finder.get_lr())
        hf.create_dataset("loss_lr", data=lr_finder.get_loss())
        hf.close()

    def augment_data(self, y):
        # flip the data
        y_flip = -1 * y
        # circular shift the data
        y_cirshift = np.roll(y, np.random.randint(1, y.shape[1] - 1), axis=1)
        return np.concatenate([y, y_flip, y_cirshift], axis=0)

    def check_speed(self, y_train):
        num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)
        y_val = y_train[indices[partial_train_num:], :, :]

        partial_y_train_original = y_train[indices[:partial_train_num], :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        # compile model with loss (weighted loss)
        self.compile()

        print("start training.")
        fit_start_time = time.time()

        if self.lambda_trainable:
            num_batches = np.ceil(
                partial_y_train.shape[0] / self.trainer.get_batch_size()
            )
            val_loss = []
            train_loss = []
            val_lambda_loss = []
            train_lambda_loss = []
            print(
                "Train on %i samples, validate on %i samples"
                % (partial_y_train.shape[0], y_val.shape[0])
            )

            # this is something to do for compile (before start training)
            self.autoencoder._make_train_function()
            self.model_2._make_train_function()

            for epoch in range(self.trainer.get_num_epochs()):
                print("epoch %s/%s" % (epoch + 1, self.trainer.get_num_epochs()))
                batches = np.linspace(0, num_batches - 1, num_batches)
                random.shuffle(batches)
                for batch in batches:
                    batch_begin_index = np.int(batch * self.trainer.get_batch_size())
                    batch_end_index = np.int(
                        (batch + 1) * self.trainer.get_batch_size()
                    )
                    training_batch_i = partial_y_train[
                        batch_begin_index:batch_end_index, :, :
                    ]

                    # train on single batch
                    self.train_on_batch(training_batch_i)

                    # update lambda on autoencoder
                    self.update_lambda()
                    # update H on model 2
                    if not self.lambda_EM:
                        self.update_H()

                    # update noiseSTD
                    if self.noiseSTD_trainable:
                        self.set_noiseSTD(noiseSTD_estimate)
        else:
            history = self.autoencoder_for_train.fit(
                partial_y_train,
                partial_y_train,
                epochs=np.int(
                    self.trainer.get_num_epochs() / self.trainer.get_num_train_reset()
                ),
                batch_size=self.trainer.get_batch_size(),
                validation_data=(y_val, y_val),
                verbose=0,
                shuffle=True,
            )

        fit_time = time.time() - fit_start_time
        print("finish training.")
        print("fit_time:", fit_time / 60, "min")

        return fit_time

    def train(self, y_train):
        num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)
        y_val = y_train[indices[partial_train_num:], :, :]

        partial_y_train_original = y_train[indices[:partial_train_num], :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        print("start training.")
        fit_start_time = time.time()

        # compile model with loss (weighted loss)
        self.compile()
        # fit (train)
        history = self.fit(partial_y_train, y_val, lr_finder, num_epochs)

        fit_time = time.time() - fit_start_time
        self.trainer.set_fit_time(fit_time)
        print("finish training.")
        print("fit_time:", fit_time / 60, "min")

        # set hisotry
        self.trainer.set_history(history)
        # set all h epochs
        self.update_H_epochs()
        if self.lambda_trainable:
            # set all lambda and nosieSTD epochs
            self.update_lambda_epochs()
            self.update_noiseSTD_epochs()
        # set the trained weights in autoencoder
        self.update_H_after_training()

    def train_and_save(self, y_train, folder_name):
        num_train = y_train.shape[0]

        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)

        y_val = y_train[indices[partial_train_num:], :, :]

        partial_y_train_original = y_train[indices[:partial_train_num], :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        # compile model with loss (weighted loss)
        self.compile()

        print("start training.")
        fit_start_time = time.time()
        for train_i in range(self.trainer.get_num_train_reset()):
            print("train_i:", train_i)
            # fit (train)
            history = self.fit(partial_y_train, y_val)
            # set hisotry
            self.trainer.set_history(history)
            # set all h epochs
            self.update_H_epochs()
            if self.lambda_trainable:
                # set all lambda and nosieSTD epochs
                self.update_lambda_epochs()
                self.update_noiseSTD_epochs()
            # set the trained weights in autoencoder
            self.update_H_after_training()
            # save results
            if train_i == 0:
                folder_time = self.save_results(folder_name)
            else:
                self.save_results(folder_name, folder_time + "-" + str(train_i))

        fit_time = time.time() - fit_start_time
        self.trainer.set_fit_time(fit_time)
        print("finish training.")
        print("fit_time:", fit_time / 60, "min")

        return folder_time

    def save_results(self, folder_name, time=1.234):
        print("save results.")
        # get history results
        history = self.trainer.get_history()
        # get H epochs
        H_epochs = self.trainer.get_H_epochs()
        if self.lambda_trainable:
            # get lambda epochs
            lambda_epochs = self.trainer.get_lambda_epochs()
            # get noiseSTD epochs
            noiseSTD_epochs = self.trainer.get_noiseSTD_epochs()
        # get H result
        H_learned = self.get_H()
        # get lambda
        lambda_value = self.get_lambda()
        # write in h5 file
        if time == 1.234:
            time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        hf = h5py.File(
            "{}/experiments/{}/results/results_training_{}.h5".format(
                PATH, folder_name, time
            ),
            "w",
        )

        if self.trainer.get_cycleLR():
            hf.create_dataset("lr_iterations", data=self.trainer.get_lr_iterations())
        hf.create_dataset("val_loss", data=history["val_loss"])
        hf.create_dataset("train_loss", data=history["train_loss"])
        if self.lambda_trainable:
            hf.create_dataset("val_lambda_loss", data=history["val_lambda_loss"])
            hf.create_dataset("train_lambda_loss", data=history["train_lambda_loss"])
            hf.create_dataset("lambda_epochs", data=lambda_epochs)
            hf.create_dataset("noiseSTD_epochs", data=noiseSTD_epochs)

        hf.create_dataset("H_epochs", data=H_epochs)
        hf.create_dataset("H_learned", data=H_learned)
        hf.create_dataset("lambda_learned", data=lambda_value)
        hf.create_dataset("lambda_donoho", data=self.lambda_donoho)
        hf.close()

        return time


class CRsAE_2d:
    def __init__(
        self,
        input_dim,
        num_conv,
        dictionary_dim,
        num_iterations,
        L,
        twosided,
        lambda_trainable,
        MIMO,
        alpha,
        num_channels,
        noiseSTD_trainable=False,
        delta=100,
        lambda_single=False,
        noiseSTD_lr=0.01,
    ):
        """
        Initializes CRsAE 2d with model and training parameters.

        """
        # models parameters
        self.input_dim = input_dim
        self.num_conv = num_conv
        self.dictionary_dim = dictionary_dim
        self.num_iterations = num_iterations
        self.L = L
        self.twosided = twosided
        self.MIMO = MIMO
        self.Ne = (self.input_dim[0] - self.dictionary_dim[0] + 1) * (
            self.input_dim[1] - self.dictionary_dim[1] + 1
        )
        self.alpha = alpha
        self.num_channels = num_channels
        self.noiseSTD_trainable = noiseSTD_trainable
        self.data_space = 2
        self.lambda_single = lambda_single
        self.noiseSTD_lr = noiseSTD_lr

        # initialize trainer
        self.trainer = trainer()

    def build_model(self, noiseSTD):
        print("build model.")

        self.noiseSTD = noiseSTD
        # compute lambda from noise level
        lambda_only_from_noise = noiseSTD * np.sqrt(2 * np.log(self.num_conv * self.Ne))
        lambda_donoho = (
            np.zeros((self.num_conv,), dtype=np.float32) + lambda_only_from_noise
        )
        self.lambda_donoho = np.copy(lambda_donoho)

        # build model
        build_graph_start_time = time.time()
        autoencoder, encoder, decoder = self.CRsAE_2d_model(self.lambda_donoho)
        build_graph_time = time.time() - build_graph_start_time
        print("build_graph_time:", np.round(build_graph_time, 4), "s")

        # this is for training purposes with alpha
        autoencoder_for_train, temp, temp = self.CRsAE_2d_model(
            self.lambda_donoho * self.alpha
        )

        if self.lambda_trainable:
            autoencoder.get_layer("FISTA").set_weights([self.lambda_donoho])
            autoencoder_for_train.get_layer("FISTA").set_weights([self.lambda_donoho])
        else:
            autoencoder.get_layer("FISTA").set_weights([self.lambda_donoho])
            autoencoder_for_train.get_layer("FISTA").set_weights(
                [self.lambda_donoho * self.alpha]
            )

        self.encoder = encoder
        self.autoencoder = autoencoder
        self.decoder = decoder
        self.autoencoder_for_train = autoencoder_for_train

        # initialize H
        self.initialize_H()

    def CRsAE_2d_model(self):
        """
        Create DAG for constraint reccurent sparse auto-encoder (CRsAE).
        :return: (autoencoder, encoder, decoder)
        """
        y = Input(shape=(self.input_dim[0], self.input_dim[1], 1), name="y")
        HT = Conv2D(
            filters=self.num_conv,
            kernel_size=self.dictionary_dim,
            padding="valid",
            use_bias=False,
            activation=None,
            trainable=True,
            input_shape=(self.input_dim[0], self.input_dim[1], 1),
            name="HT",
            kernel_constraint=max_norm(max_value=1),
        )
        # HTy
        HTy = HT(y)
        # initialize z0 to be 0 vector
        z0 = Lambda(lambda x: x * 0, name="initialize_z")(HTy)

        # Zero-pad layer
        padding_layer = ZeroPadding2D(
            padding=(
                (self.dictionary_dim[0] - 1, self.dictionary_dim[0] - 1),
                (self.dictionary_dim[1] - 1, self.dictionary_dim[1] - 1),
            ),
            name="zeropad",
        )
        # Have to define the transpose layer after creating a flowgraph with HT
        H = Conv2DFlip(HT, name="H")
        # FISTA layer
        FISTA_layer = FISTA_2d(
            HT,
            y,
            self.L,
            self.lambda_trainable,
            self.twosided,
            self.num_iterations,
            self.lambda_single,
            self.lambda_EM,
            name="FISTA",
        )

        # perform FISTA for num_iterations
        if self.lambda_trainable:
            zt, lambda_term = FISTA_layer(z0)
            lambda_placeholder = Lambda(
                lambda x: x[:, 0, 0, :] * 0 + lambda_term, name="loglambda"
            )(z0)
            lambda_zt = Lambda(lambda x: x * lambda_term, name="l1_norm")(zt)
        else:
            zt = FISTA_layer(z0)
        zt_padded = padding_layer(zt)
        # reconstruct y
        y_hat = H(zt_padded)

        encoder = Model(y, zt)

        if self.lambda_trainable:
            autoencoder = Model(y, [y_hat, lambda_zt, lambda_placeholder])
        else:
            autoencoder = Model(y, y_hat)

        # for decoding
        input_code = Input(
            shape=(
                self.input_dim[0] - self.dictionary_dim[0] + 1,
                self.input_dim[1] - self.dictionary_dim[1] + 1,
                self.num_conv,
            ),
            name="input_code",
        )
        input_code_padded = padding_layer(input_code)
        decoded = H(input_code_padded)

        decoder = Model(input_code, decoded)

        return autoencoder, encoder, decoder

    def initialize_H(self):
        self.H = np.random.randn(
            self.dictionary_dim[0], self.dictionary_dim[1], 1, self.num_conv
        )
        self.H /= np.linalg.norm(self.H, "fro", axis=(0, 1))
        # set H for autoencoder and encoder
        self.autoencoder.get_layer("HT").set_weights([self.H])
        self.autoencoder_for_train.get_layer("HT").set_weights([self.H])

    def set_H(self, H):
        self.H = H
        self.H /= np.linalg.norm(self.H, "fro", axis=(0, 1))
        # set HT in autoencoder
        self.autoencoder.get_layer("HT").set_weights([self.H])
        self.autoencoder_for_train.get_layer("HT").set_weights([self.H])

    def set_lambda(self, lambda_value):
        self.autoencoder.get_layer("FISTA").set_weights([lambda_value])
        self.autoencoder_for_train.get_layer("FISTA").set_weights([lambda_value])

    def get_H(self):
        return self.autoencoder.get_layer("HT").get_weights()[0]

    def get_H_estimate(self):
        return self.autoencoder.get_layer("HT").get_weights()[0]

    def get_lambda(self):
        lambda_value = self.autoencoder.get_layer("FISTA").get_weights()[0]
        return lambda_value

    def get_lambda_estimate(self):
        lambda_value = self.model_2.get_layer("FISTA").get_weights()[0]
        return lambda_value

    def get_input_dim(self):
        return self.input_dim

    def get_num_conv(self):
        return self.num_conv

    def get_dictionary_dim(self):
        return self.dictionary_dim

    def get_num_iterations(self):
        return self.num_iterations

    def get_L(self):
        return self.L

    def get_twosided(self):
        return self.twosided

    def get_MIMO(self):
        return self.MIMO

    def get_alpha(self):
        return self.alpha

    def get_num_channels(self):
        return self.num_channels

    def get_data_space(self):
        return self.data_space

    def update_H_after_training(self):
        if self.lambda_trainable:
            # load parameters from the best val_loss
            self.autoencoder.load_weights(
                "weights_{}.hdf5".format(self.trainer.get_unique_number())
            )
            self.set_H(self.autoencoder.get_layer("HT").get_weights()[0])
        else:
            # load parameters from the best val_loss
            self.autoencoder_for_train.load_weights(
                "weights_{}.hdf5".format(self.trainer.get_unique_number())
            )
            self.set_H(self.autoencoder_for_train.get_layer("HT").get_weights()[0])

    def update_H_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.H_epochs = []
        for epoch in range(num_epochs):
            if self.lambda_trainable:
                self.autoencoder.load_weights(
                    "weights-improvement-{}-{}.hdf5".format(
                        self.trainer.get_unique_number(), epoch + 1
                    )
                )
                self.trainer.H_epochs.append(
                    self.autoencoder.get_layer("HT").get_weights()[0]
                )
            else:
                self.autoencoder_for_train.load_weights(
                    "weights-improvement-{}-{}.hdf5".format(
                        self.trainer.get_unique_number(), epoch + 1
                    )
                )
                self.trainer.H_epochs.append(
                    self.autoencoder_for_train.get_layer("HT").get_weights()[0]
                )

    def update_lambda(self):
        lambda_value = self.get_lambda_estimate()
        self.autoencoder.get_layer("FISTA").set_weights([lambda_value])

    def encode(self, y):
        return self.encoder.predict(y)

    def decode(self, z):
        return self.decoder.predict(z)

    def separate(self, y):
        # get z
        z = self.encode(y)

        y_hat_separate = np.zeros(
            (y.shape[0], self.input_dim[0], self.input_dim[1], self.num_conv)
        )
        temp = np.zeros(
            (
                z.shape[0],
                self.input_dim[0] - self.dictionary_dim[0] + 1,
                self.input_dim[1] - self.dictionary_dim[1] + 1,
                self.num_conv,
            )
        )
        for n in range(self.num_conv):
            temp[:, :, :, n] = np.copy(z[:, :, :, n])
            decoded = np.copy(self.decode(temp))
            y_hat_separate[:, :, :, n] = np.squeeze(np.copy(decoded), axis=3)
        return y_hat_separate

    def denoise(self, y):
        return self.autoencoder.predict(y)

    def compile(self):
        def log_lambda_loss_function(y_true, y_pred):
            return K.sum(K.log(y_pred), axis=-1)

        def lambda_prior_loss_function(y_true, y_pred):
            return K.sum(K.square(y_pred - y_true), axis=-1)

        loss = self.trainer.get_loss()
        if self.lambda_trainable:
            loss_model2 = ["mae", log_lambda_loss_function, lambda_prior_loss_function]

        if self.lambda_trainable:
            # this is to weight the log lambda loss
            log_lambda_loss_weight = -self.Ne * (self.noiseSTD ** 2)
            lambda_mse_loss_weight = 1 / (self.lambda_uncertainty ** 2)

            self.autoencoder.compile(
                optimizer=self.trainer.optimizer.get_keras_optimizer(),
                loss=loss,
                loss_weights=[1 / 2],
            )
            self.model_2.compile(
                optimizer=self.trainer.optimizer.get_keras_optimizer(),
                loss=loss_model2,
                loss_weights=[1, log_lambda_loss_weight, lambda_mse_loss_weight],
            )
        else:
            self.autoencoder_for_train.compile(
                optimizer=self.trainer.optimizer.get_keras_optimizer(), loss=loss
            )

    def train_on_batch(self, training_batch_i):
        # single gradient update for H
        x, y, sample_weights = self.autoencoder._standardize_user_data(
            training_batch_i, training_batch_i, sample_weight=None, class_weight=None
        )
        if self.autoencoder._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [1.0]
        else:
            ins = x + y + sample_weights
        output_1 = self.autoencoder.train_function(ins)

        # single gradient update for lambda
        x, y, sample_weights = self.model_2._standardize_user_data(
            training_batch_i,
            [
                np.zeros(
                    (
                        training_batch_i.shape[0],
                        self.input_dim[0] - self.dictionary_dim[0] + 1,
                        self.input_dim[1] - self.dictionary_dim[1] + 1,
                        self.num_conv,
                    )
                ),
                np.zeros((training_batch_i.shape[0], self.num_conv)),
                np.zeros((training_batch_i.shape[0], self.num_conv))
                + self.lambda_donoho,
            ],
            sample_weight=None,
            class_weight=None,
        )
        if self.model_2._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [1.0]
        else:
            ins = x + y + sample_weights
        output_2 = self.model_2.train_function(ins)

    def fit(self, partial_y_train, y_val, lr_finder=[], num_epochs=4):
        if self.lambda_trainable:
            num_batches = np.ceil(
                partial_y_train.shape[0] / self.trainer.get_batch_size()
            )
            val_loss = []
            train_loss = []
            val_l1_norm_loss = []
            loglambda_loss = []
            lambda_prior_loss = []
            train_l1_norm_loss = []
            print(
                "Train on %i samples, validate on %i samples"
                % (partial_y_train.shape[0], y_val.shape[0])
            )
            for epoch in range(self.trainer.get_num_epochs()):
                batches = np.linspace(0, num_batches - 1, num_batches)
                random.shuffle(batches)
                for batch in batches:
                    batch_begin_index = np.int(batch * self.trainer.get_batch_size())
                    batch_end_index = np.int(
                        (batch + 1) * self.trainer.get_batch_size()
                    )
                    training_batch_i = partial_y_train[
                        batch_begin_index:batch_end_index, :, :, :
                    ]

                    # train on single batch
                    self.train_on_batch(training_batch_i)
                    # update lambda on autoencoder
                    self.update_lambda()
                    # update H on model 2
                    self.update_H()

                    # keep track of weighs for all epochs
                    self.autoencoder.save_weights(
                        "weights-improvement-%i-%i.hdf5"
                        % (self.trainer.get_unique_number(), epoch + 1),
                        overwrite=True,
                    )

                # test on validation set
                y_val_hat = self.denoise(y_val)
                val_loss_i = np.mean(
                    np.mean(np.square(np.squeeze(y_val - y_val_hat)), axis=1), axis=0
                )
                # get training error
                partial_y_train_hat = self.denoise(partial_y_train)
                train_loss_i = np.mean(
                    np.mean(
                        np.square(np.squeeze(partial_y_train - partial_y_train_hat)),
                        axis=1,
                    ),
                    axis=0,
                )
                # test lambda related error (weighted total, lambda x, log lambda, lambda prior)
                [val_lambda_x, log_lambda, lambda_prior] = self.model_2.predict(y_val)
                val_l1_norm_loss_i = np.mean(np.abs(val_lambda_x))
                loglambda_loss_i = np.mean(np.sum(np.log(log_lambda), axis=-1), axis=0)
                lambda_prior_loss_i = np.mean(
                    np.sum(np.square(lambda_prior - self.lambda_donoho), axis=-1),
                    axis=-0,
                )
                # get training lambda related error (weighted total, lambda x, log lambda, lambda prior)
                [train_lambda_x, log_lambda, lambda_prior] = self.model_2.predict(
                    partial_y_train
                )
                train_l1_norm_loss_i = np.mean(np.abs(train_lambda_x))

                val_loss.append(val_loss_i)
                val_l1_norm_loss.append(val_l1_norm_loss_i)
                loglambda_loss.append(loglambda_loss_i)
                lambda_prior_loss.append(lambda_prior_loss_i)
                train_loss.append(train_loss_i)
                train_l1_norm_loss.append(train_l1_norm_loss_i)

                print(
                    "Epoch %d/%d" % (epoch + 1, self.trainer.get_num_epochs()),
                    "loss:",
                    np.round(train_loss_i, 5),
                    "l1_norm_loss:",
                    np.round(train_l1_norm_loss_i, 5),
                    "val_loss:",
                    np.round(val_loss_i, 5),
                    "val_l1_norm_loss:",
                    np.round(val_l1_norm_loss_i, 5),
                    "loglambda_loss:",
                    np.round(loglambda_loss_i, 5),
                    "lambda_prior_loss:",
                    np.round(lambda_prior_loss_i, 5),
                    "weighted_loglambda_loss:",
                    np.round(-self.Ne * (self.noiseSTD ** 2) * loglambda_loss_i, 5),
                    "weighted_lambda_prior_loss:",
                    np.round(
                        1 / (self.lambda_uncertainty ** 2) * lambda_prior_loss_i, 5
                    ),
                )
                if epoch == 0:
                    min_val_loss = val_loss_i
                else:
                    min_val_loss = min(val_loss)
                if val_loss_i <= min_val_loss:
                    self.autoencoder.save_weights(
                        "weights_{}.hdf5".format(self.trainer.get_unique_number()),
                        overwrite=True,
                    )
                    if epoch == 0:
                        print(
                            "val_loss improved from inf to %.5f saving model to %s"
                            % (
                                val_loss_i,
                                "weights_{}.hdf5".format(
                                    self.trainer.get_unique_number()
                                ),
                            )
                        )
                    else:
                        print(
                            "val_loss improved from %.5f to %.5f saving model to %s"
                            % (
                                val_loss[-2],
                                val_loss_i,
                                "weights_{}.hdf5".format(
                                    self.trainer.get_unique_number()
                                ),
                            )
                        )
                else:
                    print("val_loss NOT improved.")

            history = {
                "val_loss": val_loss,
                "val_l1_norm_loss": val_l1_norm_loss,
                "loglambda_loss": loglambda_loss,
                "lambda_prior_loss": lambda_prior_loss,
                "train_loss": train_loss,
                "train_l1_norm_loss": train_l1_norm_loss,
            }
        else:
            if lr_finder:
                history = self.autoencoder_for_train.fit(
                    partial_y_train,
                    partial_y_train,
                    epochs=num_epochs,
                    batch_size=self.trainer.get_batch_size(),
                    validation_data=(y_val, y_val),
                    verbose=self.trainer.get_verbose(),
                    shuffle=True,
                    callbacks=[lr_finder],
                )
            else:
                history = self.autoencoder_for_train.fit(
                    partial_y_train,
                    partial_y_train,
                    epochs=np.int(
                        self.trainer.get_num_epochs()
                        / self.trainer.get_num_train_reset()
                    ),
                    batch_size=self.trainer.get_batch_size(),
                    validation_data=(y_val, y_val),
                    verbose=self.trainer.get_verbose(),
                    shuffle=True,
                    callbacks=self.trainer.get_callbacks(),
                )
        return history

    def find_lr(self, y_train, folder_name, num_epochs=4, min_lr=1e-7, max_lr=1e-1):
        print("find lr.")

        if self.MIMO:
            num_train = y_train[0].shape[0]
        else:
            num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)
        if self.MIMO:
            y_val = []
            partial_y_train = []
            for ch in range(self.num_channels):
                y_val.append(y_train[ch][indices[partial_train_num:], :, :, :])

                partial_y_train_original = y_train[ch][
                    indices[:partial_train_num], :, :, :
                ]
                if self.trainer.get_augment():
                    partial_y_train.append(self.augment_data(partial_y_train_original))
                else:
                    partial_y_train.append(partial_y_train_original)
        else:
            y_val = y_train[indices[partial_train_num:], :, :, :]
            partial_y_train_original = y_train[indices[:partial_train_num], :, :, :]
            if self.trainer.get_augment():
                partial_y_train = self.augment_data(partial_y_train_original)
            else:
                partial_y_train = partial_y_train_original

        # lr callback
        epoch_size = int(0.9 * num_train)
        lr_finder = LRFinder(
            min_lr=min_lr,
            max_lr=max_lr,
            steps_per_epoch=np.ceil(epoch_size / self.trainer.get_batch_size()),
            epochs=num_epochs,
        )
        # compile model with loss (weighted loss)
        self.compile()
        # fit (train)
        history = self.fit(partial_y_train, y_val, lr_finder)

        # save lr results
        hf = h5py.File(
            "../experiments/{}/results/results_lr.h5".format(folder_name), "w"
        )
        hf.create_dataset("iterations", data=lr_finder.get_iterations())
        hf.create_dataset("lr", data=lr_finder.get_lr())
        hf.create_dataset("loss_lr", data=lr_finder.get_loss())
        hf.close()

    def augment_data(self, y):
        # flip the data
        y_flip = -1 * y
        # circular shift the data
        # y_cirshift = np.roll(y,np.random.randint(1,y.shape[1]-1),axis=1)
        print(np.random.randint(1, y.shape[1]))
        # return np.concatenate([y,y_flip,y_cirshift],axis=0)
        return np.concatenate([y, y_flip], axis=0)

    def train(self, y_train):
        if self.MIMO:
            num_train = y_train[0].shape[0]
        else:
            num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)
        if self.MIMO:
            y_val = []
            partial_y_train = []
            for ch in range(self.num_channels):
                y_val.append(y_train[ch][indices[partial_train_num:], :, :, :])

                partial_y_train_original = y_train[ch][
                    indices[:partial_train_num], :, :, :
                ]
                if self.trainer.get_augment():
                    partial_y_train.append(self.augment_data(partial_y_train_original))
                else:
                    partial_y_train.append(partial_y_train_original)
        else:
            y_val = y_train[indices[partial_train_num:], :, :, :]
            partial_y_train_original = y_train[indices[:partial_train_num], :, :, :]
            if self.trainer.get_augment():
                partial_y_train = self.augment_data(partial_y_train_original)
            else:
                partial_y_train = partial_y_train_original

        print("start training.")
        fit_start_time = time.time()

        # compile model with loss (weighted loss)
        self.compile()
        # fit (train)
        history = self.fit(partial_y_train, y_val)

        fit_time = time.time() - fit_start_time
        self.trainer.set_fit_time(fit_time)
        print("finish training.")
        print("fit_time:", fit_time / 60, "min")
        # set hisotry
        self.trainer.set_history(history)
        # set all h epochs
        self.update_H_epochs()
        # set the trained weights in autoencoder
        self.update_H_after_training()

    def save_results(self, folder_name, time=1.234):
        print("save results.")
        # get history results
        history = self.trainer.get_history()
        # get H epochs
        H_epochs = self.trainer.get_H_epochs()
        # get H result
        H_learned = self.get_H()
        # get lambda
        lambda_value = self.get_lambda()
        # write in h5 file
        if time == 1.234:
            time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        hf = h5py.File(
            "{}/experiments/{}/results/results_training_{}.h5".format(
                PATH, folder_name, time
            ),
            "w",
        )
        hf.create_dataset("val_loss", data=history["val_loss"])
        hf.create_dataset("train_loss", data=history["train_loss"])
        if self.lambda_trainable:
            hf.create_dataset("val_l1_norm_loss", data=history["val_l1_norm_loss"])
            hf.create_dataset("loglambda_loss", data=history["loglambda_loss"])
            hf.create_dataset("lambda_prior_loss", data=history["lambda_prior_loss"])
            hf.create_dataset("train_l1_norm_loss", data=history["train_l1_norm_loss"])

        hf.create_dataset("H_epochs", data=H_epochs)
        hf.create_dataset("H_learned", data=H_learned)
        hf.create_dataset("lambda_learned", data=lambda_value)
        hf.create_dataset("lambda_donoho", data=self.lambda_donoho)
        hf.close()

        return time
