"""
Copyright (c) 2018 CRISP

functions to extract experiment results save in results*.h5.

:author: Bahareh Tolooshams
"""
import warnings

warnings.filterwarnings("ignore")

import click
import yaml
import h5py
import numpy as np
import fnmatch
import os
import sys
import itertools
import matplotlib.pyplot as plt

sys.path.append("..")
PATH = sys.path[-1]

from src.models.CRsAE import *
from src.prints.parameters import *
from src.plotter.plot_experiment_results import *
from src.run_experiments.extract_results_helpers import *


@click.group(chain=True)
def extract_results():
    pass


@extract_results.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def real(folder_name):
    # load model parameters
    print("load model parameters.")
    file = open("../experiments/{}/config/config_model.yml".format(folder_name), "rb")
    config_m = yaml.load(file)
    file.close()
    # load data parameters
    print("load data parameters.")
    file = open(
        "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
    )
    config_d = yaml.load(file)
    file.close()
    # load data
    print("load data.")
    hf_data = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    if config_m["MIMO"]:
        y_test = []
        y_series = []
        for ch in range(config_m["num_channels"]):
            g_ch = hf_data.get("{}".format(ch))
            y_test.append(np.array(g_ch.get("y_test")))
    else:
        g_ch = hf_data.get("{}".format(config_d["ch"]))
        y_test = np.array(g_ch.get("y_test"))
    hf_data.close()

    for file in os.listdir("{}/experiments/{}/results/".format(PATH, folder_name)):
        if fnmatch.fnmatch(file, "results_training_*"):
            file_number = file[17:-3]
            print("file number:", file_number)

            # load training results
            print("load training results.")
            hf_training = h5py.File(
                "{}/experiments/{}/results/results_training_{}.h5".format(
                    PATH, folder_name, file_number
                ),
                "r",
            )
            # load prediction results
            print("load prediction results.")
            hf_prediction = h5py.File(
                "{}/experiments/{}/results/results_prediction_{}.h5".format(
                    PATH, folder_name, file_number
                ),
                "r",
            )

            if config_m["MIMO"]:
                y_test_hat = []
                y_test_hat_separate = []
                z_test_hat = []
                for ch in range(config_m["num_channels"]):
                    g_ch = hf_prediction.get("{}".format(ch))
                    y_test_hat.append(np.array(g_ch.get("y_test_hat")))
                    z_test_hat.append(np.array(g_ch.get("z_test_hat")))
                    y_test_hat_separate.append(
                        np.array(g_ch.get("y_test_hat_separate"))
                    )
                    H_init = np.array(g_ch.get(("H_init")))
            else:
                g_ch = hf_prediction.get("{}".format(config_d["ch"]))
                y_test_hat = np.array(g_ch.get("y_test_hat"))
                z_test_hat = np.array(g_ch.get("z_test_hat"))
                y_test_hat_separate = np.array(g_ch.get("y_test_hat_separate"))
                H_init = np.array(g_ch.get(("H_init")))

            lr_iterations = np.array(hf_training.get("lr_iterations"))
            val_loss = np.array(hf_training.get("val_loss"))
            train_loss = np.array(hf_training.get("train_loss"))
            if config_m["lambda_trainable"]:
                val_l1_norm_loss = np.array(hf_training.get("val_l1_norm_loss"))
                loglambda_loss = np.array(hf_training.get("loglambda_loss"))
                lambda_prior_loss = np.array(hf_training.get("lambda_prior_loss"))
                train_l1_norm_loss = np.array(hf_training.get("train_l1_norm_loss"))
            monitor_val_loss = np.array(hf_training.get(config_m["loss_type"]))
            H_epochs = np.array(hf_training.get("H_epochs"))
            H_learned = np.array(hf_training.get("H_learned"))
            lambda_donoho = np.array(hf_training.get("lambda_donoho"))
            lambda_learned = np.array(hf_training.get("lambda_learned"))

            hf_training.close()
            hf_prediction.close()
            ################################################
            # get distance error of the dictionary
            dist_init_learned, best_permutation_index = get_err_h1_h2(H_init, H_learned)

            num_conv = H_epochs.shape[-1]
            num_epochs = H_epochs.shape[0]
            dist_init_learned_epochs = np.zeros((num_conv, num_epochs))
            for epoch in range(num_epochs):
                dist_init_learned_epochs[:, epoch], temp = get_err_h1_h2(
                    H_init, H_epochs[epoch, :, :, :], best_permutation_index
                )
            ################################################
            best_val_epoch = np.argmin(monitor_val_loss)
            best_epoch = np.min(best_val_epoch)

            plot_loss(
                val_loss,
                train_loss,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            if config_m["cycleLR"]:
                plot_lr_iterations(
                    lr_iterations,
                    val_loss.shape[0],
                    PATH,
                    folder_name,
                    file_number,
                    line_width=2,
                    scale=1.2,
                    scale_height=1,
                    text_font=20,
                    title_font=20,
                    axes_font=20,
                    legend_font=20,
                    number_font=20,
                )

            if config_m["lambda_trainable"]:
                plot_lambda(
                    lambda_init,
                    lambda_epochs,
                    best_epoch,
                    best_val_epoch,
                    PATH,
                    folder_name,
                    file_number,
                    row=1,
                    line_width=2,
                    marker_size=30,
                    scale=4,
                    scale_height=0.5,
                    text_font=45,
                    title_font=55,
                    axes_font=48,
                    legend_font=34,
                    number_font=40,
                )

                plot_noiseSTD(
                    noiseSTD_epochs,
                    best_epoch,
                    best_val_epoch,
                    PATH,
                    folder_name,
                    file_number,
                    line_width=2,
                    marker_size=30,
                    scale=4,
                    scale_height=0.5,
                    text_font=45,
                    title_font=55,
                    axes_font=48,
                    legend_font=34,
                    number_font=40,
                )

                plot_lambda_loss(
                    val_lambda_loss,
                    train_lambda_loss,
                    best_epoch,
                    best_val_epoch,
                    PATH,
                    folder_name,
                    file_number,
                    line_width=2,
                    marker_size=15,
                    scale=1.2,
                    scale_height=1,
                    text_font=20,
                    title_font=20,
                    axes_font=20,
                    legend_font=20,
                    number_font=20,
                )

            plot_H_err_epochs_real(
                dist_init_learned_epochs,
                best_epoch,
                PATH,
                folder_name,
                file_number,
                y_fine=0.5,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            # plot dictionary
            plot_H_real(
                H_init,
                H_learned,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                y_fine=0.5,
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_denoise_real(
                0,
                y_test,
                y_test_hat,
                config_m["MIMO"],
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_code_real(
                0,
                z_test_hat,
                config_m["MIMO"],
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_H_epochs_real(
                H_init,
                H_learned,
                H_epochs,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                y_fine=0.5,
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            # RMSE
            if y_test.shape[0] == 1:
                RMSE_y_yhat_test = np.sqrt(
                    np.mean(
                        np.power((np.squeeze(y_test) - np.squeeze(y_test_hat)), 2),
                        axis=0,
                    )
                )
            else:
                RMSE_y_yhat_test = np.mean(
                    np.sqrt(
                        np.mean(
                            np.power((np.squeeze(y_test) - np.squeeze(y_test_hat)), 2),
                            axis=1,
                        )
                    )
                )

            # l1 norm
            l1_norm_z_test_hat = np.mean(np.sum(np.abs(z_test_hat), axis=1), axis=0)

            summary = {
                "distance error init learned": np.round(dist_init_learned, 3).tolist(),
                "averaged distance error init learned": np.mean(
                    np.round(dist_init_learned, 3)
                ).tolist(),
                "noiseSTD": np.round(noiseSTD, 3).tolist(),
                "RMSE test": np.round(RMSE_y_yhat_test, 3).tolist(),
                "l1 norm test estimated code": np.round(l1_norm_z_test_hat, 3).tolist(),
                "lambda donoho": np.round(lambda_donoho, 5).tolist(),
                "lambda learned": np.round(lambda_learned, 5).tolist(),
            }

            with open(
                "{}/experiments/{}/reports/summary_{}.yaml".format(
                    PATH, folder_name, file_number
                ),
                "w",
            ) as outfile:
                yaml.dump(summary, outfile, default_flow_style=False)


@extract_results.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def real_series(folder_name):
    # load model parameters
    print("load model parameters.")
    file = open("../experiments/{}/config/config_model.yml".format(folder_name), "rb")
    config_m = yaml.load(file)
    file.close()
    # load data parameters
    print("load data parameters.")
    file = open(
        "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
    )
    config_d = yaml.load(file)
    file.close()
    # load data
    print("load data.")
    hf_data = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    if config_m["MIMO"]:
        y_test = []
        y_series = []
        for ch in range(config_m["num_channels"]):
            g_ch = hf_data.get("{}".format(ch))
            y_test.append(np.array(g_ch.get("y_test")))
            y_series.append(np.array(g_ch.get("y_series")))
    else:
        g_ch = hf_data.get("{}".format(config_d["ch"]))
        y_test = np.array(g_ch.get("y_test"))
        y_series = np.array(g_ch.get("y_series"))
    hf_data.close()
    # load spikes
    print("load spikes.")
    spike_channel = config_d["ch"]
    hf_data = h5py.File(
        "{}/experiments/{}/data/spikes.h5".format(PATH, folder_name), "r"
    )
    g_ch = hf_data.get("{}".format(spike_channel))
    spikes = np.array(g_ch.get("spikes"))
    hf_data.close()

    all_missed_list = []
    all_false_list = []
    for file in os.listdir("{}/experiments/{}/results/".format(PATH, folder_name)):
        if fnmatch.fnmatch(file, "results_training_*"):
            # skip files related to multiple val shuffle of the same training
            if file[-5] == "-":
                continue

            file_number = file[17:-3]
            print("file number:", file_number)

            H_epochs = []
            best_val_epoch = []
            for train_i in range(config_m["num_train_reset"]):
                # load training results
                if train_i == 0:
                    hf_training = h5py.File(
                        "{}/experiments/{}/results/results_training_{}.h5".format(
                            PATH, folder_name, file_number
                        ),
                        "r",
                    )
                    # load prediction results
                    hf_prediction = h5py.File(
                        "{}/experiments/{}/results/results_prediction_{}.h5".format(
                            PATH, folder_name, file_number
                        ),
                        "r",
                    )
                    if config_m["MIMO"]:
                        y_test_hat = []
                        z_test_hat = []
                        y_test_hat_separate = []
                        y_series_hat = []
                        z_series_hat = []
                        y_series_hat_separate = []
                        for ch in range(config_m["num_channels"]):
                            g_ch = hf_prediction.get("{}".format(ch))
                            y_test_hat.append(np.array(g_ch.get("y_test_hat")))
                            z_test_hat.append(np.array(g_ch.get("z_test_hat")))
                            y_test_hat_separate.append(
                                np.array(g_ch.get("y_test_hat_separate"))
                            )
                            y_series_hat.append(np.array(g_ch.get("y_series_hat")))
                            z_series_hat.append(np.array(g_ch.get("z_series_hat")))
                            y_series_hat_separate.append(
                                np.array(g_ch.get("y_series_hat_separate"))
                            )
                            H_init = np.array(g_ch.get(("H_init")))
                    else:
                        g_ch = hf_prediction.get("{}".format(config_d["ch"]))
                        y_test_hat = np.array(g_ch.get("y_test_hat"))
                        z_test_hat = np.array(g_ch.get("z_test_hat"))
                        y_test_hat_separate = np.array(g_ch.get("y_test_hat_separate"))
                        y_series_hat = np.array(g_ch.get("y_series_hat"))
                        z_series_hat = np.array(g_ch.get("z_series_hat"))
                        y_series_hat_separate = np.array(
                            g_ch.get("y_series_hat_separate")
                        )
                        H_init = np.array(g_ch.get(("H_init")))
                else:
                    hf_training = h5py.File(
                        "{}/experiments/{}/results/results_training_{}-{}.h5".format(
                            PATH, folder_name, file_number, train_i
                        ),
                        "r",
                    )

                lr_iterations = np.array(hf_training.get("lr_iterations"))
                val_loss = np.array(hf_training.get("val_loss"))
                if config_m["lambda_trainable"]:
                    val_l1_norm_loss = np.array(hf_training.get("val_l1_norm_loss"))
                    loglambda_loss = np.array(hf_training.get("loglambda_loss"))
                    lambda_prior_loss = np.array(hf_training.get("lambda_prior_loss"))
                    train_l1_norm_loss = np.array(hf_training.get("train_l1_norm_loss"))
                monitor_val_loss = np.array(hf_training.get(config_m["loss_type"]))
                train_loss = np.array(hf_training.get("train_loss"))
                H_epochs = np.array(hf_training.get("H_epochs"))
                H_learned = np.array(hf_training.get("H_learned"))
                lambda_donoho = np.array(hf_training.get("lambda_donoho"))
                lambda_learned = np.array(hf_training.get("lambda_learned"))

                if train_i == 0:
                    val_loss = np.array(hf_training.get("val_loss"))
                    train_loss = np.array(hf_training.get("train_loss"))
                    if config_m["lambda_trainable"]:
                        val_lambda_loss = np.array(hf_training.get("val_lambda_loss"))
                        train_lambda_loss = np.array(
                            hf_training.get("train_lambda_loss")
                        )

                    monitor_val_loss = np.array(hf_training.get(config_m["loss_type"]))
                    H_epochs = np.array(hf_training.get("H_epochs"))
                    best_val_epoch.append(np.argmin(monitor_val_loss))
                else:
                    val_loss = np.concatenate(
                        [val_loss, np.array(hf_training.get("val_loss"))]
                    )
                    train_loss = np.concatenate(
                        [train_loss, np.array(hf_training.get("train_loss"))]
                    )
                    if config_m["lambda_trainable"]:
                        val_lambda_loss = np.concatenate(
                            [
                                val_lambda_loss,
                                np.array(hf_training.get("val_lambda_loss")),
                            ]
                        )
                        train_lambda_loss = np.concatenate(
                            [
                                train_lambda_loss,
                                np.array(hf_training.get("train_lambda_loss")),
                            ]
                        )

                    H_epochs = np.concatenate(
                        [H_epochs, np.array(hf_training.get("H_epochs"))], axis=0
                    )
                    best_val_epoch.append(
                        np.argmin(np.array(hf_training.get(config_m["loss_type"])))
                        + train_i
                        * np.int(config_m["num_epochs"] / config_m["num_train_reset"])
                    )

                H_learned = np.array(hf_training.get("H_learned"))

                hf_training.close()
                hf_prediction.close()
            ################################################
            # get distance error of the dictionary
            dist_init_learned, best_permutation_index = get_err_h1_h2(H_init, H_learned)

            num_conv = H_epochs.shape[-1]
            num_epochs = H_epochs.shape[0]
            dist_init_learned_epochs = np.zeros((num_conv, num_epochs))
            for epoch in range(num_epochs):
                dist_init_learned_epochs[:, epoch], temp = get_err_h1_h2(
                    H_init, H_epochs[epoch, :, :, :], best_permutation_index
                )
            ################################################
            # get miss-false result and plot
            spikes_channel = config_d["spikes_channel"]
            event_range = config_d["event_range"]
            th_list = np.arange(15, 85, 2)

            # for n in range(config_m["num_conv"]):
            #     spikes_filter = n
            #     if config_m["MIMO"]:
            #         for ch in range(config_m["num_channels"]):
            #             if ch == 0:
            #                 th_list = np.arange(5, 120, 5)
            #             else:
            #                 th_list = np.arange(5, 70, 5)
            #
            #             y_series_hat_conv = y_series_hat_separate[ch][
            #                 :, :, spikes_filter
            #             ]
            #
            #             z_conv = np.expand_dims(np.copy(spikes), axis=0)
            #             missed_events, missed_list, false_events, false_list = get_miss_false(
            #                 z_conv,
            #                 y_series_hat_conv,
            #                 spikes_filter,
            #                 th_list,
            #                 event_range,
            #             )
            #
            #             all_missed_list.append(missed_list)
            #             all_false_list.append(false_list)
            #
            #             plot_miss_false(
            #                 missed_list,
            #                 false_list,
            #                 PATH,
            #                 folder_name,
            #                 file_number,
            #                 spikes_filter,
            #                 ch,
            #                 line_width=2,
            #                 marker_size=15,
            #                 scale=1.2,
            #                 scale_height=1,
            #                 text_font=20,
            #                 title_font=20,
            #                 axes_font=20,
            #                 legend_font=20,
            #                 number_font=20,
            #             )
            #
            #             plot_H_and_miss_false(
            #                 H_init,
            #                 H_learned,
            #                 missed_list,
            #                 false_list,
            #                 PATH,
            #                 folder_name,
            #                 file_number,
            #                 spikes_filter,
            #                 ch,
            #                 config_d["sampling_rate"],
            #                 line_width=2.5,
            #                 marker_size=30,
            #                 scale=4,
            #                 scale_height=0.5,
            #                 text_font=38,
            #                 title_font=50,
            #                 axes_font=40,
            #                 legend_font=35,
            #                 number_font=35,
            #             )
            #     else:
            #         th_list = np.arange(5, 120, 5)
            #         y_series_hat_conv = y_series_hat_separate[:, :, spikes_filter]
            #
            #         z_conv = np.expand_dims(np.copy(spikes), axis=0)
            #         missed_events, missed_list, false_events, false_list = get_miss_false(
            #             z_conv,
            #             y_series_hat_conv,
            #             spikes_filter,
            #             th_list,
            #             event_range,
            #         )
            #
            #         all_missed_list.append(missed_list)
            #         all_false_list.append(false_list)
            #
            #         plot_miss_false(
            #             missed_list,
            #             false_list,
            #             PATH,
            #             folder_name,
            #             file_number,
            #             spikes_filter,
            #             config_d["ch"],
            #             line_width=2,
            #             marker_size=15,
            #             scale=1.2,
            #             scale_height=1,
            #             text_font=20,
            #             title_font=20,
            #             axes_font=20,
            #             legend_font=20,
            #             number_font=20,
            #         )
            #
            #         plot_H_and_miss_false(
            #             H_init,
            #             H_learned,
            #             missed_list,
            #             false_list,
            #             PATH,
            #             folder_name,
            #             file_number,
            #             spikes_filter,
            #             config_d["ch"],
            #             config_d["sampling_rate"],
            #             line_width=2.5,
            #             marker_size=30,
            #             scale=4,
            #             scale_height=0.5,
            #             text_font=38,
            #             title_font=50,
            #             axes_font=40,
            #             legend_font=35,
            #             number_font=35,
            #         )

            ################################################
            best_epoch = np.min(best_val_epoch)

            plot_separate_real_series(
                0,
                60000,
                spikes,
                y_series_hat_separate,
                config_m["MIMO"],
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                line_width=2,
                marker_size=30,
                scale=4,
                scale_height=0.5,
                text_font=30,
                title_font=30,
                axes_font=30,
                legend_font=30,
                number_font=25,
            )

            plot_loss(
                val_loss,
                train_loss,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            if config_m["cycleLR"]:
                plot_lr_iterations(
                    lr_iterations,
                    val_loss.shape[0],
                    PATH,
                    folder_name,
                    file_number,
                    line_width=2,
                    scale=1.2,
                    scale_height=1,
                    text_font=20,
                    title_font=20,
                    axes_font=20,
                    legend_font=20,
                    number_font=20,
                )

            if config_m["lambda_trainable"]:
                plot_lambda(
                    lambda_init,
                    lambda_epochs,
                    best_epoch,
                    best_val_epoch,
                    PATH,
                    folder_name,
                    file_number,
                    row=1,
                    line_width=2,
                    marker_size=30,
                    scale=4,
                    scale_height=0.5,
                    text_font=45,
                    title_font=55,
                    axes_font=48,
                    legend_font=34,
                    number_font=40,
                )

                plot_noiseSTD(
                    noiseSTD_epochs,
                    best_epoch,
                    best_val_epoch,
                    PATH,
                    folder_name,
                    file_number,
                    line_width=2,
                    marker_size=30,
                    scale=4,
                    scale_height=0.5,
                    text_font=45,
                    title_font=55,
                    axes_font=48,
                    legend_font=34,
                    number_font=40,
                )

                plot_lambda_loss(
                    val_lambda_loss,
                    train_lambda_loss,
                    best_epoch,
                    best_val_epoch,
                    PATH,
                    folder_name,
                    file_number,
                    line_width=2,
                    marker_size=15,
                    scale=1.2,
                    scale_height=1,
                    text_font=20,
                    title_font=20,
                    axes_font=20,
                    legend_font=20,
                    number_font=20,
                )

            plot_H_err_epochs_real(
                dist_init_learned_epochs,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                y_fine=0.5,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            # plot dictionary
            plot_H_real(
                H_init,
                H_learned,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                y_fine=0.5,
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            if config_m["MIMO"]:
                scale = 4
                scale_height = 1
            else:
                scale = 4
                scale_height = 0.5

            plot_denoise_real(
                0,
                y_test,
                y_test_hat,
                config_m["MIMO"],
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                line_width=2,
                marker_size=15,
                scale=scale,
                scale_height=scale_height,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_code_real(
                0,
                z_test_hat,
                config_m["MIMO"],
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                line_width=2,
                marker_size=15,
                scale=scale,
                scale_height=scale_height,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_H_epochs_real(
                H_init,
                H_learned,
                H_epochs,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                y_fine=0.5,
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            # RMSE
            if config_m["MIMO"]:
                num_channels = config_m["num_channels"]
                RMSE_y_yhat_test = []
                l1_norm_z_test_hat = []
                for ch in range(num_channels):
                    if y_test[ch].shape[0] == 1:
                        RMSE_y_yhat_test.append(
                            np.sqrt(
                                np.mean(
                                    np.power(
                                        (
                                            np.squeeze(y_test[ch])
                                            - np.squeeze(y_test_hat[ch])
                                        ),
                                        2,
                                    ),
                                    axis=0,
                                )
                            )
                        )
                    else:
                        RMSE_y_yhat_test.append(
                            np.mean(
                                np.sqrt(
                                    np.mean(
                                        np.power(
                                            (
                                                np.squeeze(y_test[ch])
                                                - np.squeeze(y_test_hat[ch])
                                            ),
                                            2,
                                        ),
                                        axis=1,
                                    )
                                )
                            )
                        )

                    # l1 norm
                    l1_norm_z_test_hat.append(
                        np.mean(np.sum(np.abs(z_test_hat[ch]), axis=1), axis=0)
                    )
            else:
                if y_test.shape[0] == 1:
                    RMSE_y_yhat_test = np.sqrt(
                        np.mean(
                            np.power((np.squeeze(y_test) - np.squeeze(y_test_hat)), 2),
                            axis=0,
                        )
                    )
                else:
                    RMSE_y_yhat_test = np.mean(
                        np.sqrt(
                            np.mean(
                                np.power(
                                    (np.squeeze(y_test) - np.squeeze(y_test_hat)), 2
                                ),
                                axis=1,
                            )
                        )
                    )

                # l1 norm
                l1_norm_z_test_hat = np.mean(np.sum(np.abs(z_test_hat), axis=1), axis=0)

            summary = {
                "distance error init learned": np.round(dist_init_learned, 3).tolist(),
                "averaged distance error init learned": np.mean(
                    np.round(dist_init_learned, 3)
                ).tolist(),
                "noiseSTD": np.round(config_d["noiseSTD"], 3).tolist(),
                "RMSE test": np.round(RMSE_y_yhat_test, 3).tolist(),
                "l1 norm test estimated code": np.round(l1_norm_z_test_hat, 3).tolist(),
                "lambda donoho": np.round(lambda_donoho, 5).tolist(),
                "lambda learned": np.round(lambda_learned, 5).tolist(),
            }

            with open(
                "{}/experiments/{}/reports/summary_{}.yaml".format(
                    PATH, folder_name, file_number
                ),
                "w",
            ) as outfile:
                yaml.dump(summary, outfile, default_flow_style=False)

    plot_all_miss_false(
        all_missed_list,
        all_false_list,
        PATH,
        folder_name,
        line_width=2,
        marker_size=15,
        scale=1.2,
        scale_height=1,
        text_font=20,
        title_font=20,
        axes_font=20,
        legend_font=20,
        number_font=20,
    )


@extract_results.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def simulated(folder_name):
    # load model parameters
    print("load model parameters.")
    file = open("../experiments/{}/config/config_model.yml".format(folder_name), "rb")
    config_m = yaml.load(file)
    file.close()
    # load data parameters
    print("load data parameters.")
    file = open(
        "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
    )
    config_d = yaml.load(file)
    file.close()
    # load data
    print("load data.")
    hf_data = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    if config_m["MIMO"]:
        print("Error: This script is only for single channel model")
    else:
        y_train = np.array(hf_data.get("y_train"))
        y_test = np.array(hf_data.get("y_test"))
        y_train_noisy = np.array(hf_data.get("y_train_noisy"))
        y_test_noisy = np.array(hf_data.get("y_test_noisy"))
        z_test = np.array(hf_data.get("z_test"))
        noiseSTD = np.array(hf_data.get("noiseSTD"))
    hf_data.close()

    for file in os.listdir("{}/experiments/{}/results/".format(PATH, folder_name)):
        if fnmatch.fnmatch(file, "results_training_*"):
            # skip files related to multiple val shuffle of the same training
            if file[-5] == "-":
                continue

            file_number = file[17:-3]
            print("file number:", file_number)

            # load H_true
            H_true = np.load(
                "{}/experiments/{}/data/H_true.npy".format(PATH, folder_name)
            )

            H_epochs = []
            best_val_epoch = []
            for train_i in range(config_m["num_train_reset"]):
                # load training results
                if train_i == 0:
                    hf_training = h5py.File(
                        "{}/experiments/{}/results/results_training_{}.h5".format(
                            PATH, folder_name, file_number
                        ),
                        "r",
                    )
                    # load prediction results
                    hf_prediction = h5py.File(
                        "{}/experiments/{}/results/results_prediction_{}.h5".format(
                            PATH, folder_name, file_number
                        ),
                        "r",
                    )
                    if config_m["MIMO"]:
                        print("Error: This script is only for single channel model")
                    else:
                        g_ch = hf_prediction.get("{}".format(config_d["ch"]))
                        y_test_hat = np.array(g_ch.get("y_test_hat"))
                        z_test_hat = np.array(g_ch.get("z_test_hat"))
                        H_init = np.array(g_ch.get(("H_init")))
                        lambda_init = np.array(g_ch.get(("lambda_init")))
                else:
                    hf_training = h5py.File(
                        "{}/experiments/{}/results/results_training_{}-{}.h5".format(
                            PATH, folder_name, file_number, train_i
                        ),
                        "r",
                    )

                if train_i == 0:
                    if config_m["cycleLR"]:
                        lr_iterations = np.array(hf_training.get("lr_iterations"))
                    val_loss = np.array(hf_training.get("val_loss"))
                    train_loss = np.array(hf_training.get("train_loss"))
                    if config_m["lambda_trainable"]:
                        val_lambda_loss = np.array(hf_training.get("val_lambda_loss"))
                        train_lambda_loss = np.array(
                            hf_training.get("train_lambda_loss")
                        )

                    monitor_val_loss = np.array(hf_training.get(config_m["loss_type"]))
                    H_epochs = np.array(hf_training.get("H_epochs"))
                    lambda_epochs = np.array(hf_training.get("lambda_epochs"))
                    noiseSTD_epochs = np.array(hf_training.get("noiseSTD_epochs"))
                    best_val_epoch.append(np.argmin(monitor_val_loss))
                else:
                    if config_m["cycleLR"]:
                        lr_iterations = np.concatenate(
                            [lr_iterations, np.array(hf_training.get("lr_iterations"))]
                        )
                    val_loss = np.concatenate(
                        [val_loss, np.array(hf_training.get("val_loss"))]
                    )
                    train_loss = np.concatenate(
                        [train_loss, np.array(hf_training.get("train_loss"))]
                    )
                    if config_m["lambda_trainable"]:
                        val_lambda_loss = np.concatenate(
                            [
                                val_lambda_loss,
                                np.array(hf_training.get("val_lambda_loss")),
                            ]
                        )
                        train_lambda_loss = np.concatenate(
                            [
                                train_lambda_loss,
                                np.array(hf_training.get("train_lambda_loss")),
                            ]
                        )

                    H_epochs = np.concatenate(
                        [H_epochs, np.array(hf_training.get("H_epochs"))], axis=0
                    )
                    # lambda_epochs = np.squeeze(np.concatenate(
                    #     [np.expand_dims(lambda_epochs,axis=1), np.expand_dims(np.array(hf_training.get("lambda_epochs")),axis=1)],axis=0)
                    # )
                    # noiseSTD_epochs = np.squeeze(np.concatenate(
                    #     [np.expand_dims(noiseSTD_epochs,axis=1), np.expand_dims(np.array(hf_training.get("noiseSTD_epochs")),axis=1)],axis=0)
                    # )

                    best_val_epoch.append(
                        np.argmin(np.array(hf_training.get(config_m["loss_type"])))
                        + train_i
                        * np.int(config_m["num_epochs"] / config_m["num_train_reset"])
                    )

                H_learned = np.array(hf_training.get("H_learned"))
                lambda_donoho = np.array(hf_training.get("lambda_donoho"))
                lambda_learned = np.array(hf_training.get("lambda_learned"))

                hf_training.close()
                hf_prediction.close()

            ################################################
            # get distance error of the dictionary
            dist_true_learned, best_permutation_index = get_err_h1_h2(H_true, H_learned)
            dist_true_init, temp = get_err_h1_h2(H_true, H_init, best_permutation_index)

            dist_true_init_notswap, temp = get_err_h1_h2(H_true, H_init)

            H_last = H_epochs[-1, :, :, :]
            dist_true_last, best_permutation_index_last = get_err_h1_h2(H_true, H_last)
            dist_true_init_last, temp = get_err_h1_h2(
                H_true, H_init, best_permutation_index_last
            )

            num_conv = H_epochs.shape[-1]
            num_epochs = H_epochs.shape[0]
            dictionary_dim = H_epochs.shape[1]
            dist_true_learned_epochs = np.zeros((num_conv, num_epochs))
            dist_true_learned_epochs_last = np.zeros((num_conv, num_epochs))
            for epoch in range(num_epochs):
                dist_true_learned_epochs[:, epoch], temp = get_err_h1_h2(
                    H_true, H_epochs[epoch, :, :, :], best_permutation_index
                )
                dist_true_learned_epochs_last[:, epoch], temp = get_err_h1_h2(
                    H_true, H_epochs[epoch, :, :, :], best_permutation_index_last
                )
            flip = np.ones(num_conv)
            delay = np.zeros(num_conv)
            flip_last = np.ones(num_conv)
            delay_last = np.zeros(num_conv)
            permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))
            for n in range(num_conv):
                cross_corr = np.correlate(
                    H_true[:, 0, n],
                    H_learned[:, 0, permutations[best_permutation_index][n]],
                    "full",
                )
                delay[n] = dictionary_dim - np.argmax(abs(cross_corr)) - 1
                pos_corr = np.max(cross_corr)
                neg_corr = np.abs(np.min(cross_corr))

                if pos_corr < neg_corr:
                    flip[n] *= -1

                cross_corr = np.correlate(
                    H_true[:, 0, n],
                    H_last[:, 0, permutations[best_permutation_index_last][n]],
                    "full",
                )
                delay_last[n] = dictionary_dim - np.argmax(abs(cross_corr)) - 1
                pos_corr = np.max(cross_corr)
                neg_corr = np.abs(np.min(cross_corr))

                if pos_corr < neg_corr:
                    flip_last[n] *= -1

            ################################################

            best_epoch = np.min(best_val_epoch)

            plot_loss(
                np.log10(val_loss),
                np.log10(train_loss),
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            if config_m["cycleLR"]:
                plot_lr_iterations(
                    lr_iterations,
                    val_loss.shape[0],
                    PATH,
                    folder_name,
                    file_number,
                    line_width=2,
                    scale=1.2,
                    scale_height=1,
                    text_font=20,
                    title_font=20,
                    axes_font=20,
                    legend_font=20,
                    number_font=20,
                )

            if config_m["lambda_trainable"]:
                plot_lambda(
                    lambda_init,
                    lambda_epochs,
                    best_epoch,
                    best_val_epoch,
                    PATH,
                    folder_name,
                    file_number,
                    row=1,
                    line_width=2,
                    marker_size=30,
                    scale=4,
                    scale_height=0.5,
                    text_font=45,
                    title_font=55,
                    axes_font=48,
                    legend_font=34,
                    number_font=40,
                )

                plot_noiseSTD(
                    noiseSTD_epochs,
                    best_epoch,
                    best_val_epoch,
                    PATH,
                    folder_name,
                    file_number,
                    line_width=2,
                    marker_size=30,
                    scale=4,
                    scale_height=0.5,
                    text_font=45,
                    title_font=55,
                    axes_font=48,
                    legend_font=34,
                    number_font=40,
                )

                plot_lambda_loss(
                    val_lambda_loss,
                    train_lambda_loss,
                    best_epoch,
                    best_val_epoch,
                    PATH,
                    folder_name,
                    file_number,
                    line_width=2,
                    marker_size=15,
                    scale=1.2,
                    scale_height=1,
                    text_font=20,
                    title_font=20,
                    axes_font=20,
                    legend_font=20,
                    number_font=20,
                )

            plot_H_err_epochs_sim(
                np.log10(dist_true_learned_epochs),
                np.log10(dist_true_init),
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                y_fine=0.2,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            plot_H_err_epochs_sim(
                np.log10(dist_true_learned_epochs_last),
                np.log10(dist_true_init_last),
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number + "last",
                y_fine=0.2,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            plot_H_err_epochs_sim_subplot(
                np.log10(dist_true_learned_epochs),
                np.log10(dist_true_init),
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                row=1,
                y_fine=15,
                line_width=2.2,
                marker_size=15,
                scale=4,
                scale_height=0.75,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            # plot dictionary
            plot_H_sim(
                H_true,
                H_init,
                H_learned,
                best_permutation_index,
                flip,
                delay,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                row=1,
                y_fine=0.5,
                line_width=2.2,
                marker_size=15,
                scale=4,
                scale_height=0.75,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_denoise_sim(
                0,
                y_test,
                y_test_noisy,
                y_test_hat,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_code_sim(
                0,
                z_test,
                z_test_hat,
                best_permutation_index,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                row=1,
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_H_epochs_sim(
                H_true,
                H_init,
                H_learned,
                H_epochs,
                best_permutation_index,
                flip,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                row=1,
                y_fine=0.5,
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            # RMSE
            if y_test_noisy.shape[0] == 1:
                RMSE_y_yhat_test = np.sqrt(
                    np.mean(
                        np.power(
                            (np.squeeze(y_test_noisy) - np.squeeze(y_test_hat)), 2
                        ),
                        axis=0,
                    )
                )
                RMSE_ytrue_yhat_test = np.sqrt(
                    np.mean(
                        np.power((np.squeeze(y_test) - np.squeeze(y_test_hat)), 2),
                        axis=0,
                    )
                )

            else:
                RMSE_y_yhat_test = np.mean(
                    np.sqrt(
                        np.mean(
                            np.power(
                                (np.squeeze(y_test_noisy) - np.squeeze(y_test_hat)), 2
                            ),
                            axis=1,
                        )
                    )
                )
                RMSE_ytrue_yhat_test = np.mean(
                    np.sqrt(
                        np.mean(
                            np.power((np.squeeze(y_test) - np.squeeze(y_test_hat)), 2),
                            axis=1,
                        )
                    )
                )

            # l1 norm
            l1_norm_z_test = np.mean(np.sum(np.abs(z_test), axis=1), axis=0)
            l1_norm_z_test_hat = np.mean(np.sum(np.abs(z_test_hat), axis=1), axis=0)

            summary = {
                "distance error true init not swap": np.round(
                    dist_true_init_notswap, 3
                ).tolist(),
                "distance error true init": np.round(dist_true_init, 3).tolist(),
                "distance error true learned": np.round(dist_true_learned, 3).tolist(),
                "distance error true init from last": np.round(
                    dist_true_init_last, 3
                ).tolist(),
                "distance error true last": np.round(dist_true_last, 3).tolist(),
                "averaged distance error true learned": np.mean(
                    np.round(dist_true_learned, 3)
                ).tolist(),
                "averaged distance error true last": np.mean(
                    np.round(dist_true_last, 3)
                ).tolist(),
                "noiseSTD": np.round(noiseSTD, 3).tolist(),
                "RMSE test": np.round(RMSE_y_yhat_test, 3).tolist(),
                "RMSE test compared to true": np.round(
                    RMSE_ytrue_yhat_test, 3
                ).tolist(),
                "l1 norm test code": np.round(l1_norm_z_test, 3).tolist(),
                "l1 norm test estimated code": np.round(l1_norm_z_test_hat, 3).tolist(),
                "lambda donoho": np.round(lambda_donoho, 5).tolist(),
                "lambda learned": np.round(lambda_learned, 5).tolist(),
            }

            with open(
                "{}/experiments/{}/reports/summary_{}.yaml".format(
                    PATH, folder_name, file_number
                ),
                "w",
            ) as outfile:
                yaml.dump(summary, outfile, default_flow_style=False)


@extract_results.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def lcsc_simulated(folder_name):
    # load model parameters
    print("load model parameters.")
    file = open("../experiments/{}/config/config_model.yml".format(folder_name), "rb")
    config_m = yaml.load(file)
    file.close()
    # load data parameters
    print("load data parameters.")
    file = open(
        "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
    )
    config_d = yaml.load(file)
    file.close()
    # load data
    print("load data.")
    hf_data = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    if "MIMO" in config_m:
        if config_m["MIMO"]:
            print("Error: This script is only for single channel model")

    y_train = np.array(hf_data.get("y_train"))
    y_test = np.array(hf_data.get("y_test"))
    y_train_noisy = np.array(hf_data.get("y_train_noisy"))
    y_test_noisy = np.array(hf_data.get("y_test_noisy"))
    z_test = np.array(hf_data.get("z_test"))
    noiseSTD = np.array(hf_data.get("noiseSTD"))
    hf_data.close()

    for file in os.listdir("{}/experiments/{}/results/".format(PATH, folder_name)):
        if fnmatch.fnmatch(file, "results_training_*"):
            # skip files related to multiple val shuffle of the same training
            if file[-5] == "-":
                continue

            file_number = file[17:-3]
            print("file number:", file_number)

            # load H_true
            H_true = np.load(
                "{}/experiments/{}/data/H_true.npy".format(PATH, folder_name)
            )

            H_epochs = []
            best_val_epoch = []
            for train_i in range(config_m["num_train_reset"]):
                # load training results
                if train_i == 0:
                    hf_training = h5py.File(
                        "{}/experiments/{}/results/results_training_{}.h5".format(
                            PATH, folder_name, file_number
                        ),
                        "r",
                    )
                    # load prediction results
                    hf_prediction = h5py.File(
                        "{}/experiments/{}/results/results_prediction_{}.h5".format(
                            PATH, folder_name, file_number
                        ),
                        "r",
                    )
                    if "MIMO" in config_m:
                        if config_m["MIMO"]:
                            print("Error: This script is only for single channel model")

                    g_ch = hf_prediction.get("{}".format(config_d["ch"]))
                    y_test_hat = np.array(g_ch.get("y_test_hat"))
                    z_test_hat = np.array(g_ch.get("z_test_hat"))
                    Wd_init = np.array(g_ch.get(("Wd_init")))
                    We_init = np.array(g_ch.get(("We_init")))
                    d_init = np.array(g_ch.get(("d_init")))
                    lambda_init = np.array(g_ch.get(("lambda_init")))
                else:
                    hf_training = h5py.File(
                        "{}/experiments/{}/results/results_training_{}-{}.h5".format(
                            PATH, folder_name, file_number, train_i
                        ),
                        "r",
                    )

                if train_i == 0:
                    val_loss = np.array(hf_training.get("val_loss"))
                    train_loss = np.array(hf_training.get("train_loss"))
                    monitor_val_loss = np.array(hf_training.get(config_m["loss_type"]))
                    Wd_epochs = np.array(hf_training.get("Wd_epochs"))
                    We_epochs = np.array(hf_training.get("We_epochs"))
                    d_epochs = np.array(hf_training.get("d_epochs"))
                    lambda_epochs = np.array(hf_training.get("lambda_epochs"))
                    best_val_epoch.append(np.argmin(monitor_val_loss))
                else:
                    val_loss = np.concatenate(
                        [val_loss, np.array(hf_training.get("val_loss"))]
                    )
                    train_loss = np.concatenate(
                        [train_loss, np.array(hf_training.get("train_loss"))]
                    )
                    Wd_epochs = np.concatenate(
                        [Wd_epochs, np.array(hf_training.get("Wd_epochs"))], axis=0
                    )
                    We_epochs = np.concatenate(
                        [We_epochs, np.array(hf_training.get("We_epochs"))], axis=0
                    )
                    d_epochs = np.concatenate(
                        [d_epochs, np.array(hf_training.get("d_epochs"))], axis=0
                    )
                    lambda_epochs = np.concatenate(
                        [lambda_epochs, np.array(hf_training.get("lambda_epochs"))],
                        axis=0,
                    )
                    best_val_epoch.append(
                        np.argmin(np.array(hf_training.get(config_m["loss_type"])))
                        + train_i
                        * np.int(config_m["num_epochs"] / config_m["num_train_reset"])
                    )

                Wd_learned = np.array(hf_training.get("Wd_learned"))
                We_learned = np.array(hf_training.get("We_learned"))
                d_learned = np.array(hf_training.get("d_learned"))
                lambda_learned = np.array(hf_training.get("lambda_learned"))

                hf_training.close()
                hf_prediction.close()
            ################################################
            # get distance error of the weights
            dist_We_true_learned, best_We_permutation_index = get_err_h1_h2(
                H_true, We_learned
            )
            dist_We_true_init, temp = get_err_h1_h2(
                H_true, We_init, best_We_permutation_index
            )

            dist_Wd_true_learned, best_Wd_permutation_index = get_err_h1_h2(
                H_true, np.expand_dims(np.flip(np.squeeze(Wd_learned), axis=0), axis=1)
            )
            dist_Wd_true_init, temp = get_err_h1_h2(
                H_true,
                np.expand_dims(np.flip(np.squeeze(Wd_init), axis=0), axis=1),
                best_Wd_permutation_index,
            )

            dist_d_true_learned, best_d_permutation_index = get_err_h1_h2(
                H_true, np.expand_dims(np.flip(np.squeeze(d_learned), axis=0), axis=1)
            )
            dist_d_true_init, temp = get_err_h1_h2(
                H_true,
                np.expand_dims(np.flip(np.squeeze(d_init), axis=0), axis=1),
                best_d_permutation_index,
            )

            num_conv = We_epochs.shape[-1]
            num_epochs = We_epochs.shape[0]
            dictionary_dim = We_epochs.shape[1]
            dist_We_true_learned_epochs = np.zeros((num_conv, num_epochs))
            dist_Wd_true_learned_epochs = np.zeros((num_conv, num_epochs))
            dist_d_true_learned_epochs = np.zeros((num_conv, num_epochs))
            for epoch in range(num_epochs):
                dist_We_true_learned_epochs[:, epoch], temp = get_err_h1_h2(
                    H_true, We_epochs[epoch, :, :, :], best_We_permutation_index
                )
                dist_Wd_true_learned_epochs[:, epoch], temp = get_err_h1_h2(
                    H_true,
                    np.expand_dims(
                        np.flip(np.squeeze(Wd_epochs[epoch, :, :, :]), axis=0), axis=1
                    ),
                    best_Wd_permutation_index,
                )
                dist_d_true_learned_epochs[:, epoch], temp = get_err_h1_h2(
                    H_true,
                    np.expand_dims(
                        np.flip(np.squeeze(d_epochs[epoch, :, :, :]), axis=0), axis=1
                    ),
                    best_d_permutation_index,
                )
            flip = np.ones(num_conv)
            delay = np.zeros(num_conv)
            permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))
            # for n in range(num_conv):
            #     cross_corr = np.correlate(
            #         H_true[:, 0, n],
            #         H_learned[:, 0, permutations[best_permutation_index][n]],
            #         "full",
            #     )
            #     delay[n] = dictionary_dim - np.argmax(abs(cross_corr)) - 1
            #     pos_corr = np.max(cross_corr)
            #     neg_corr = np.abs(np.min(cross_corr))
            #
            #     if pos_corr < neg_corr:
            #         flip[n] *= -1

            ################################################

            best_epoch = np.min(best_val_epoch)

            plot_loss(
                val_loss,
                train_loss,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            plot_lambda(
                lambda_init,
                lambda_epochs,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                row=1,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            plot_H_err_epochs_sim(
                dist_We_true_learned_epochs,
                dist_We_true_init,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                y_fine=0.2,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
                output_name="We",
            )

            plot_H_err_epochs_sim(
                dist_Wd_true_learned_epochs,
                dist_Wd_true_init,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                y_fine=0.2,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
                output_name="Wd",
            )

            plot_H_err_epochs_sim(
                dist_d_true_learned_epochs,
                dist_d_true_init,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                y_fine=0.2,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
                output_name="d",
            )

            # plot Wd
            plot_Wd_sim(
                H_true,
                Wd_init,
                Wd_learned,
                best_Wd_permutation_index,
                flip,
                delay,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                row=1,
                y_fine=0.5,
                line_width=2.2,
                marker_size=15,
                scale=4,
                scale_height=0.75,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            # plot We
            plot_We_sim(
                H_true,
                We_init,
                We_learned,
                best_We_permutation_index,
                flip,
                delay,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                row=1,
                y_fine=0.5,
                line_width=2.2,
                marker_size=15,
                scale=4,
                scale_height=0.75,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            # plot d
            plot_d_sim(
                H_true,
                d_init,
                d_learned,
                best_d_permutation_index,
                flip,
                delay,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                row=1,
                y_fine=0.5,
                line_width=2.2,
                marker_size=15,
                scale=4,
                scale_height=0.75,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_denoise_sim(
                0,
                y_test,
                y_test_noisy,
                y_test_hat,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_code_sim(
                0,
                z_test,
                z_test_hat,
                best_d_permutation_index,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                row=1,
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            # RMSE
            if y_test_noisy.shape[0] == 1:
                RMSE_y_yhat_test = np.sqrt(
                    np.mean(
                        np.power(
                            (np.squeeze(y_test_noisy) - np.squeeze(y_test_hat)), 2
                        ),
                        axis=0,
                    )
                )
                RMSE_ytrue_yhat_test = np.sqrt(
                    np.mean(
                        np.power((np.squeeze(y_test) - np.squeeze(y_test_hat)), 2),
                        axis=0,
                    )
                )

            else:
                RMSE_y_yhat_test = np.mean(
                    np.sqrt(
                        np.mean(
                            np.power(
                                (np.squeeze(y_test_noisy) - np.squeeze(y_test_hat)), 2
                            ),
                            axis=1,
                        )
                    )
                )
                RMSE_ytrue_yhat_test = np.mean(
                    np.sqrt(
                        np.mean(
                            np.power((np.squeeze(y_test) - np.squeeze(y_test_hat)), 2),
                            axis=1,
                        )
                    )
                )

            # l1 norm
            l1_norm_z_test = np.mean(np.sum(np.abs(z_test), axis=1), axis=0)
            l1_norm_z_test_hat = np.mean(np.sum(np.abs(z_test_hat), axis=1), axis=0)

            summary = {
                "Wd distance error true init": np.round(dist_Wd_true_init, 3).tolist(),
                "Wd distance error true learned": np.round(
                    dist_Wd_true_learned, 3
                ).tolist(),
                "Wd averaged distance error true learned": np.mean(
                    np.round(dist_Wd_true_learned, 3)
                ).tolist(),
                "We distance error true init": np.round(dist_We_true_init, 3).tolist(),
                "We distance error true learned": np.round(
                    dist_We_true_learned, 3
                ).tolist(),
                "We averaged distance error true learned": np.mean(
                    np.round(dist_We_true_learned, 3)
                ).tolist(),
                "d distance error true init": np.round(dist_d_true_init, 3).tolist(),
                "d distance error true learned": np.round(
                    dist_d_true_learned, 3
                ).tolist(),
                "d averaged distance error true learned": np.mean(
                    np.round(dist_d_true_learned, 3)
                ).tolist(),
                "noiseSTD": np.round(noiseSTD, 3).tolist(),
                "RMSE test": np.round(RMSE_y_yhat_test, 8).tolist(),
                "RMSE test compared to true": np.round(
                    RMSE_ytrue_yhat_test, 8
                ).tolist(),
                "l1 norm test code": np.round(l1_norm_z_test, 3).tolist(),
                "l1 norm test estimated code": np.round(l1_norm_z_test_hat, 3).tolist(),
                "lambda learned": np.round(lambda_learned, 5).tolist(),
            }

            with open(
                "{}/experiments/{}/reports/summary_{}.yaml".format(
                    PATH, folder_name, file_number
                ),
                "w",
            ) as outfile:
                yaml.dump(summary, outfile, default_flow_style=False)


@extract_results.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def tlae_simulated(folder_name):
    # load model parameters
    print("load model parameters.")
    file = open("../experiments/{}/config/config_model.yml".format(folder_name), "rb")
    config_m = yaml.load(file)
    file.close()
    # load data parameters
    print("load data parameters.")
    file = open(
        "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
    )
    config_d = yaml.load(file)
    file.close()
    # load data
    print("load data.")
    hf_data = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    if config_m["MIMO"]:
        print("Error: This script is only for single channel model")
    else:
        y_train = np.array(hf_data.get("y_train"))
        y_test = np.array(hf_data.get("y_test"))
        y_train_noisy = np.array(hf_data.get("y_train_noisy"))
        y_test_noisy = np.array(hf_data.get("y_test_noisy"))
        z_test = np.array(hf_data.get("z_test"))
        noiseSTD = np.array(hf_data.get("noiseSTD"))
    hf_data.close()

    for file in os.listdir("{}/experiments/{}/results/".format(PATH, folder_name)):
        if fnmatch.fnmatch(file, "TLAE_results_training_*"):
            # skip files related to multiple val shuffle of the same training
            if file[-5] == "-":
                continue

            file_number = file[22:-3]
            print("file number:", file_number)

            # load H_true
            H_true = np.load(
                "{}/experiments/{}/data/H_true.npy".format(PATH, folder_name)
            )

            H_epochs = []
            best_val_epoch = []
            for train_i in range(config_m["num_train_reset"]):
                # load training results
                if train_i == 0:
                    hf_training = h5py.File(
                        "{}/experiments/{}/results/TLAE_results_training_{}.h5".format(
                            PATH, folder_name, file_number
                        ),
                        "r",
                    )
                    # load prediction results
                    hf_prediction = h5py.File(
                        "{}/experiments/{}/results/TLAE_results_prediction_{}.h5".format(
                            PATH, folder_name, file_number
                        ),
                        "r",
                    )
                    if config_m["MIMO"]:
                        print("Error: This script is only for single channel model")
                    else:
                        g_ch = hf_prediction.get("{}".format(config_d["ch"]))
                        z_test_hat = np.array(g_ch.get("z_test_hat"))
                        H_init = np.array(g_ch.get(("H_init")))
                        lambda_init = np.array(g_ch.get(("lambda_init")))
                else:
                    hf_training = h5py.File(
                        "{}/experiments/{}/results/TLAE_results_training_{}-{}.h5".format(
                            PATH, folder_name, file_number, train_i
                        ),
                        "r",
                    )

                if train_i == 0:
                    val_loss = np.array(hf_training.get("val_loss"))
                    train_loss = np.array(hf_training.get("train_loss"))

                    monitor_val_loss = np.array(hf_training.get(config_m["loss_type"]))
                    H_epochs = np.array(hf_training.get("H_epochs"))
                    best_val_epoch.append(np.argmin(monitor_val_loss))
                else:
                    val_loss = np.concatenate(
                        [val_loss, np.array(hf_training.get("val_loss"))]
                    )
                    train_loss = np.concatenate(
                        [train_loss, np.array(hf_training.get("train_loss"))]
                    )

                    H_epochs = np.concatenate(
                        [H_epochs, np.array(hf_training.get("H_epochs"))], axis=0
                    )

                    best_val_epoch.append(
                        np.argmin(np.array(hf_training.get(config_m["loss_type"])))
                        + train_i
                        * np.int(config_m["num_epochs"] / config_m["num_train_reset"])
                    )

                H_learned = np.array(hf_training.get("H_learned"))
                lambda_donoho = np.array(hf_training.get("lambda_donoho"))
                lambda_learned = np.array(hf_training.get("lambda_learned"))

                hf_training.close()
                hf_prediction.close()
            ################################################
            # get distance error of the dictionary
            dist_true_learned, best_permutation_index = get_err_h1_h2(H_true, H_learned)
            best_permutation_index = 0
            dist_true_init, temp = get_err_h1_h2(H_true, H_init, best_permutation_index)

            num_conv = H_epochs.shape[-1]
            num_epochs = H_epochs.shape[0]
            dictionary_dim = H_epochs.shape[1]
            dist_true_learned_epochs = np.zeros((num_conv, num_epochs))
            for epoch in range(num_epochs):
                dist_true_learned_epochs[:, epoch], temp = get_err_h1_h2(
                    H_true, H_epochs[epoch, :, :, :], best_permutation_index
                )
            flip = np.ones(num_conv)
            delay = np.zeros(num_conv)
            permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))
            # for n in range(num_conv):
            #     cross_corr = np.correlate(
            #         H_true[:, 0, n],
            #         H_learned[:, 0, permutations[best_permutation_index][n]],
            #         "full",
            #     )
            #     delay[n] = dictionary_dim - np.argmax(abs(cross_corr)) - 1
            #     pos_corr = np.max(cross_corr)
            #     neg_corr = np.abs(np.min(cross_corr))
            #
            #     if pos_corr < neg_corr:
            #         flip[n] *= -1

            ################################################

            best_epoch = np.min(best_val_epoch)

            plot_loss(
                val_loss,
                train_loss,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            plot_H_err_epochs_sim(
                dist_true_learned_epochs,
                dist_true_init,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                y_fine=0.2,
                line_width=2,
                marker_size=15,
                scale=1.2,
                scale_height=1,
                text_font=20,
                title_font=20,
                axes_font=20,
                legend_font=20,
                number_font=20,
            )

            plot_H_err_epochs_sim_subplot(
                dist_true_learned_epochs,
                dist_true_init,
                best_epoch,
                best_val_epoch,
                PATH,
                folder_name,
                file_number,
                row=1,
                y_fine=15,
                line_width=2.2,
                marker_size=15,
                scale=4,
                scale_height=0.75,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            # plot dictionary
            plot_H_sim(
                H_true,
                H_init,
                H_learned,
                best_permutation_index,
                flip,
                delay,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                row=1,
                y_fine=0.5,
                line_width=2.2,
                marker_size=15,
                scale=4,
                scale_height=0.75,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_code_sim(
                0,
                z_test,
                z_test_hat,
                best_permutation_index,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                row=1,
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            plot_H_epochs_sim(
                H_true,
                H_init,
                H_learned,
                H_epochs,
                best_permutation_index,
                flip,
                PATH,
                folder_name,
                file_number,
                config_d["sampling_rate"],
                row=1,
                y_fine=0.5,
                line_width=2,
                marker_size=15,
                scale=4,
                scale_height=0.5,
                text_font=45,
                title_font=45,
                axes_font=48,
                legend_font=32,
                number_font=40,
            )

            # # RMSE
            # if y_test_noisy.shape[0] == 1:
            #     RMSE_y_yhat_test = np.sqrt(
            #         np.mean(
            #             np.power(
            #                 (np.squeeze(y_test_noisy) - np.squeeze(y_test_hat)), 2
            #             ),
            #             axis=0,
            #         )
            #     )
            #     RMSE_ytrue_yhat_test = np.sqrt(
            #         np.mean(
            #             np.power((np.squeeze(y_test) - np.squeeze(y_test_hat)), 2),
            #             axis=0,
            #         )
            #     )
            #
            # else:
            #     RMSE_y_yhat_test = np.mean(
            #         np.sqrt(
            #             np.mean(
            #                 np.power(
            #                     (np.squeeze(y_test_noisy) - np.squeeze(y_test_hat)), 2
            #                 ),
            #                 axis=1,
            #             )
            #         )
            #     )
            #     RMSE_ytrue_yhat_test = np.mean(
            #         np.sqrt(
            #             np.mean(
            #                 np.power((np.squeeze(y_test) - np.squeeze(y_test_hat)), 2),
            #                 axis=1,
            #             )
            #         )
            #     )
            #
            # # l1 norm
            # l1_norm_z_test = np.mean(np.sum(np.abs(z_test), axis=1), axis=0)
            # l1_norm_z_test_hat = np.mean(np.sum(np.abs(z_test_hat), axis=1), axis=0)
            #
            # summary = {
            #     "distance error true init": np.round(dist_true_init, 3).tolist(),
            #     "distance error true learned": np.round(dist_true_learned, 3).tolist(),
            #     "averaged distance error true learned": np.mean(
            #         np.round(dist_true_learned, 3)
            #     ).tolist(),
            #     "noiseSTD": np.round(noiseSTD, 3).tolist(),
            #     "RMSE test": np.round(RMSE_y_yhat_test, 3).tolist(),
            #     "RMSE test compared to true": np.round(
            #         RMSE_ytrue_yhat_test, 3
            #     ).tolist(),
            #     "l1 norm test code": np.round(l1_norm_z_test, 3).tolist(),
            #     "l1 norm test estimated code": np.round(l1_norm_z_test_hat, 3).tolist(),
            #     "lambda donoho": np.round(lambda_donoho, 5).tolist(),
            #     "lambda learned": np.round(lambda_learned, 5).tolist(),
            # }
            #
            # with open(
            #     "{}/experiments/{}/reports/summary_{}.yaml".format(
            #         PATH, folder_name, file_number
            #     ),
            #     "w",
            # ) as outfile:
            #     yaml.dump(summary, outfile, default_flow_style=False)


@extract_results.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def loss_crsae_vs_sporco(folder_name):
    # load model parameters
    print("load model parameters.")
    file = open("../experiments/{}/config/config_model.yml".format(folder_name), "rb")
    config_m = yaml.load(file)
    file.close()
    # load data parameters
    print("load data parameters.")
    file = open(
        "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
    )
    config_d = yaml.load(file)
    file.close()

    crsae_file_number = "2018-11-20-01-28-49"
    sporco_file_number = "2018-11-20-00-23-05"

    hf_training = h5py.File(
        "{}/experiments/{}/results/results_training_{}.h5".format(
            PATH, folder_name, crsae_file_number
        ),
        "r",
    )

    sporco_hf_training = h5py.File(
        "{}/experiments/{}/results/results_sporco_{}.h5".format(
            PATH, folder_name, sporco_file_number
        ),
        "r",
    )

    # load H_true
    H_true = np.load("{}/experiments/{}/data/H_true.npy".format(PATH, folder_name))

    val_loss = np.array(hf_training.get("val_loss"))
    train_loss = np.array(hf_training.get("train_loss"))
    crsae_fit_time = np.array(hf_training.get("fit_time"))
    crsae_fit_time = 3.530170941352844 * 60
    hf_training.close()

    sporco_l2_loss = np.array(sporco_hf_training.get("l2_loss"))
    # get rmse from loss
    sporco_l2_loss *= 2 / (
        config_d["num_train"] * config_m["val_split"] * config_m["input_dim"]
    )

    sporco_fit_time = np.array(sporco_hf_training.get("fit_time"))
    hf_training.close()

    best_epoch = np.min(val_loss)

    plot_loss_crsae_vs_sporco(
        np.log10(train_loss),
        np.log10(sporco_l2_loss),
        crsae_fit_time,
        sporco_fit_time,
        PATH,
        folder_name,
        crsae_file_number,
        sporco_file_number,
        row=1,
        y_fine=0.5,
        line_width=6,
        marker_size=15,
        scale=4,
        scale_height=0.75,
        text_font=45,
        title_font=45,
        axes_font=48,
        legend_font=32,
        number_font=40,
    )


@extract_results.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def lr_loss(folder_name):
    plot_lr_loss(
        PATH,
        folder_name,
        line_width=2,
        scale=1.2,
        scale_height=1,
        text_font=20,
        title_font=20,
        axes_font=20,
        legend_font=20,
        number_font=20,
    )


@extract_results.command()
@click.option("--folder_names", help="list of folder names in experiment directory")
def rmse_vs_alpha(folder_names):
    RMSE = []
    RMSE_noisy = []
    dist_err = []
    alpha_list = []
    for folder_name in folder_names:
        print(folder_name)
        # load model parameters
        file = open(
            "../experiments/{}/config/config_model.yml".format(folder_name), "rb"
        )
        config_m = yaml.load(file)
        file.close()
        # load data parameters
        file = open(
            "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
        )
        config_d = yaml.load(file)
        file.close()
        # load data
        hf_data = h5py.File(
            "{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r"
        )
        if config_m["MIMO"]:
            print("Error: This script is only for single channel model")
        else:
            y_test = np.array(hf_data.get("y_test"))
            y_test_noisy = np.array(hf_data.get("y_test_noisy"))
            z_test = np.array(hf_data.get("z_test"))
            noiseSTD = np.array(hf_data.get("noiseSTD"))
        hf_data.close()

        RMSE_cur_alpha_list = []
        RMSE_noisy_cur_alpha_list = []
        dist_err_cur_alpha_list = []
        for file in os.listdir("{}/experiments/{}/results/".format(PATH, folder_name)):
            if fnmatch.fnmatch(file, "results_training_*"):
                file_number = file[17:-3]
                print("file number:", file_number)

                # load training results
                hf_training = h5py.File(
                    "{}/experiments/{}/results/results_training_{}.h5".format(
                        PATH, folder_name, file_number
                    ),
                    "r",
                )
                # load prediction results
                hf_prediction = h5py.File(
                    "{}/experiments/{}/results/results_prediction_{}.h5".format(
                        PATH, folder_name, file_number
                    ),
                    "r",
                )
                # load H_true
                H_true = np.load(
                    "{}/experiments/{}/data/H_true.npy".format(PATH, folder_name)
                )

                if config_m["MIMO"]:
                    print("Error: This script is only for single channel model")
                else:
                    g_ch = hf_prediction.get("{}".format(config_d["ch"]))
                    z_test_hat = np.array(g_ch.get("z_test_hat"))
                    y_test_hat = np.array(g_ch.get("y_test_hat"))

                H_learned = np.array(hf_training.get("H_learned"))

                hf_training.close()
                hf_prediction.close()
                ################################################
                # get distance error of the dictionary
                dist_true_learned, best_permutation_index = get_err_h1_h2(
                    H_true, H_learned
                )
                ################################################

                # RMSE
                if y_test_noisy.shape[0] == 1:
                    RMSE_y_yhat_test = np.sqrt(
                        np.mean(
                            np.power(
                                (np.squeeze(y_test_noisy) - np.squeeze(y_test_hat)), 2
                            ),
                            axis=0,
                        )
                    )
                    RMSE_ytrue_yhat_test = np.sqrt(
                        np.mean(
                            np.power((np.squeeze(y_test) - np.squeeze(y_test_hat)), 2),
                            axis=0,
                        )
                    )

                else:
                    RMSE_y_yhat_test = np.mean(
                        np.sqrt(
                            np.mean(
                                np.power(
                                    (np.squeeze(y_test_noisy) - np.squeeze(y_test_hat)),
                                    2,
                                ),
                                axis=1,
                            )
                        )
                    )
                    RMSE_ytrue_yhat_test = np.mean(
                        np.sqrt(
                            np.mean(
                                np.power(
                                    (np.squeeze(y_test) - np.squeeze(y_test_hat)), 2
                                ),
                                axis=1,
                            )
                        )
                    )

                # append to the list
                RMSE_cur_alpha_list.append(RMSE_ytrue_yhat_test)
                RMSE_noisy_cur_alpha_list.append(RMSE_y_yhat_test)
                dist_err_cur_alpha_list.append(np.mean(dist_true_learned))

        RMSE_cur_alpha = np.mean(RMSE_cur_alpha_list, axis=0)
        RMSE_noisy_cur_alpha = np.mean(RMSE_noisy_cur_alpha_list, axis=0)
        dist_err_cur_alpha = np.mean(dist_err_cur_alpha_list, axis=0)

        RMSE.append(RMSE_cur_alpha)
        RMSE_noisy.append(RMSE_noisy_cur_alpha)
        dist_err.append(dist_err_cur_alpha)
        alpha_list.append(config_m["alpha"])

    plot_alpha_tune_sim(
        alpha_list,
        RMSE,
        RMSE_noisy,
        dist_err,
        PATH,
        folder_name,
        file_number,
        line_width=2,
        marker_size=15,
        scale=1.2,
        scale_height=1,
        text_font=20,
        title_font=20,
        axes_font=20,
        legend_font=10,
        number_font=15,
    )


@extract_results.command()
@click.option("--folder_names", help="list of folder names in experiment directory")
def snr_vs_err(folder_names):
    snr_list = []
    dist_err = []
    dist_init_err = []
    for folder_name in folder_names:
        # load model parameters
        file = open(
            "../experiments/{}/config/config_model.yml".format(folder_name), "rb"
        )
        config_m = yaml.load(file)
        file.close()
        # load data parameters
        file = open(
            "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
        )
        config_d = yaml.load(file)
        file.close()

        print("SNR:", config_d["snr"])

        dist_err_cur_snr_list = []
        dist_init_err_cur_snr_list = []
        for file in os.listdir("{}/experiments/{}/results/".format(PATH, folder_name)):
            if fnmatch.fnmatch(file, "results_training_*"):
                file_number = file[17:-3]
                print("file number:", file_number)

                # load training results
                hf_training = h5py.File(
                    "{}/experiments/{}/results/results_training_{}.h5".format(
                        PATH, folder_name, file_number
                    ),
                    "r",
                )
                # load prediction results
                hf_prediction = h5py.File(
                    "{}/experiments/{}/results/results_prediction_{}.h5".format(
                        PATH, folder_name, file_number
                    ),
                    "r",
                )
                if config_m["MIMO"]:
                    print("Error: This script is only for single channel model")
                else:
                    g_ch = hf_prediction.get("{}".format(config_d["ch"]))
                    H_init = np.array(g_ch.get(("H_init")))
                # load H_true
                H_true = np.load(
                    "{}/experiments/{}/data/H_true.npy".format(PATH, folder_name)
                )

                H_learned = np.array(hf_training.get("H_learned"))

                hf_training.close()
                hf_prediction.close()
                ################################################
                # get distance error of the dictionary
                dist_true_learned, best_permutation_index = get_err_h1_h2(
                    H_true, H_learned
                )

                dist_true_init, temp = get_err_h1_h2(
                    H_true, H_init, best_permutation_index
                )
                ################################################
                dist_err_cur_snr_list.append(dist_true_learned)
                dist_init_err_cur_snr_list.append(dist_true_init)

        dist_err_cur_snr = np.mean(dist_err_cur_snr_list, axis=0)
        dist_init_err_cur_snr = np.mean(dist_init_err_cur_snr_list, axis=0)

        dist_err.append(dist_err_cur_snr)
        dist_init_err.append(dist_init_err_cur_snr)
        snr_list.append(config_d["snr"])

    plot_snr_results(
        snr_list,
        dist_err,
        dist_init_err,
        PATH,
        folder_name,
        file_number,
        y_fine=0.2,
        line_width=2,
        marker_size=15,
        scale=1.2,
        scale_height=1,
        text_font=20,
        title_font=20,
        axes_font=20,
        legend_font=20,
        number_font=20,
    )


@extract_results.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def fwd_alpha(folder_name):
    # load model parameters
    print("load model parameters.")
    file = open("../experiments/{}/config/config_model.yml".format(folder_name), "rb")
    config_m = yaml.load(file)
    file.close()
    # load data parameters
    print("load data parameters.")
    file = open(
        "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
    )
    config_d = yaml.load(file)
    file.close()
    # load data
    print("load data.")
    hf_data = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    if config_m["MIMO"]:
        print("Error: This script is only for single channel model")
    else:
        y_train = np.array(hf_data.get("y_train"))
        y_test = np.array(hf_data.get("y_test"))
        y_train_noisy = np.array(hf_data.get("y_train_noisy"))
        y_test_noisy = np.array(hf_data.get("y_test_noisy"))
        z_test = np.array(hf_data.get("z_test"))
        noiseSTD = np.array(hf_data.get("noiseSTD"))
    hf_data.close()

    for file in os.listdir("{}/experiments/{}/results/".format(PATH, folder_name)):
        if fnmatch.fnmatch(file, "results_fwd_*"):
            file_number = file[12:-3]
            print("file number:", file_number)

            # load fwd results
            hf_fwd = h5py.File(
                "{}/experiments/{}/results/results_fwd_{}.h5".format(
                    PATH, folder_name, file_number
                ),
                "r",
            )
            if config_m["MIMO"]:
                print("Error: This script is only for single channel model")
            else:
                g_ch = hf_fwd.get("{}".format(config_d["ch"]))
                alpha_list = np.array(g_ch.get("alpha_list"))
                RMSE_y_yhat_test_list = []
                RMSE_ytrue_yhat_test_list = []
                ctr = 0
                for alpha in alpha_list:
                    y_test_hat = np.array(g_ch.get("y_test_hat_{}".format(ctr)))

                    # RMSE
                    if y_test_noisy.shape[0] == 1:
                        RMSE_y_yhat_test = np.sqrt(
                            np.mean(
                                np.power(
                                    (np.squeeze(y_test_noisy) - np.squeeze(y_test_hat)),
                                    2,
                                ),
                                axis=0,
                            )
                        )
                        RMSE_ytrue_yhat_test = np.sqrt(
                            np.mean(
                                np.power(
                                    (np.squeeze(y_test) - np.squeeze(y_test_hat)), 2
                                ),
                                axis=0,
                            )
                        )
                    else:
                        RMSE_y_yhat_test = np.mean(
                            np.sqrt(
                                np.mean(
                                    np.power(
                                        (
                                            np.squeeze(y_test_noisy)
                                            - np.squeeze(y_test_hat)
                                        ),
                                        2,
                                    ),
                                    axis=1,
                                )
                            )
                        )
                        RMSE_ytrue_yhat_test = np.mean(
                            np.sqrt(
                                np.mean(
                                    np.power(
                                        (np.squeeze(y_test) - np.squeeze(y_test_hat)), 2
                                    ),
                                    axis=1,
                                )
                            )
                        )
                    ctr += 1
                    RMSE_y_yhat_test_list.append(RMSE_y_yhat_test)
                    RMSE_ytrue_yhat_test_list.append(RMSE_ytrue_yhat_test)
                hf_fwd.close()

                print(noiseSTD)
                plot_fwd_alpha_vs_rmse(
                    alpha_list,
                    RMSE_ytrue_yhat_test_list,
                    RMSE_y_yhat_test_list,
                    noiseSTD,
                    PATH,
                    folder_name,
                    file_number,
                    line_width=2,
                    marker_size=15,
                    scale=1.2,
                    scale_height=1,
                    text_font=20,
                    title_font=20,
                    axes_font=15,
                    legend_font=20,
                    number_font=20,
                )


if __name__ == "__main__":
    extract_results()
