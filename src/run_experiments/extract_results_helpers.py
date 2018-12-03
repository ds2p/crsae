"""
Copyright (c) 2018 CRISP

helper functions for extract_results.py

:author: Bahareh Tolooshams
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import itertools


def get_err_h1_h2(h1, h2, best_permutation_index=-1.234):
    """
    Helper function to get distance error between dictionary h1 and h2
    :param h1
    :param h2
    :return: (err_dist, best_permutation_index)
    """
    # normalize h
    h1 /= np.linalg.norm(h1, axis=0)
    h2 /= np.linalg.norm(h2, axis=0)
    num_conv = h1.shape[2]
    dictionary_dim = h1.shape[0]

    permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))
    if best_permutation_index == -1.234:
        err_dist_all = np.zeros((num_conv, len(permutations)))
        # loop over all permutations
        for per in range(len(permutations)):
            # loop over each convolutional dicitonary
            for n in range(num_conv):
                # get cross-correlation
                cross_corr = np.correlate(
                    h1[:, 0, n], h2[:, 0, permutations[per][n]], "full"
                )
                dot = np.max(np.abs(cross_corr))
                err_dist_all[n, per] = np.sqrt(1 - (np.power(dot, 2)))
        best_permutation_index = np.argmin(np.mean(err_dist_all, axis=0))
        err_dist = err_dist_all[:, best_permutation_index]
    else:
        err_dist = np.zeros(num_conv)
        # loop over each convolutional dicitonary
        for n in range(num_conv):
            # get cross-correlation
            cross_corr = np.correlate(
                h1[:, 0, n], h2[:, 0, permutations[best_permutation_index][n]], "full"
            )
            dot = np.max(np.abs(cross_corr))
            err_dist[n] = np.sqrt(1 - (np.power(dot, 2)))

    return err_dist, best_permutation_index


def get_miss_false(z_conv, y_hat_conv, conv, th_list, event_range):
    """
    Helper function to extract miss-false spike info
    :param z_conv: true sparse code
    :param y_hat_conv: reconstruction of conv dictionary separately
    :param threshold: predict an event for z_hat beyond the threshold
    :param event_range: range of which spike appearance is matched with the true
    :return: (missed_per, false_per)
    """
    missed_list = []
    false_list = []

    for th in th_list:
        events, events_hat = event_prediction(z_conv, y_hat_conv, conv, th, event_range)
        missed_events, missed_per, false_events, false_per = miss_false_event(
            events, events_hat, event_range, conv
        )
        missed_list.append(missed_per)
        false_list.append(false_per)
    return missed_events, missed_list, false_events, false_list


def event_prediction(z_conv, y_hat_conv, conv, threshold, event_range):
    """
    Helper function to predict spike (sparse code)
    :param z_conv: true sparse code for conv filter
    :param y_hat_conv: reconstruction of conv dictionary separately
    :param threshold: predict an event for z_hat beyond the threshold
    :param event_range: range of which spike appearance is matched with the true
    :return: (events, events_hat)
    """
    num_data = z_conv.shape[0]
    z_dim = z_conv.shape[1]

    events = []
    events_hat = []
    for i in range(num_data):
        event = {}
        event_hat = {}
        zin = np.copy(z_conv[i, :])
        yin_hat = np.copy(y_hat_conv[i, :])
        event["h{}".format(conv + 1)] = np.where(zin != 0)[0]

        spikes_hat_i = np.copy(yin_hat)

        # count as spike if yin_hat is above threshold
        spikes_hat_i[np.where(abs(yin_hat) >= threshold)] = 1
        spikes_hat_i[np.where(abs(yin_hat) < threshold)] = 0

        # count as only one spike, for values of 1 within the event_range
        for k in range(len(yin_hat)):
            if spikes_hat_i[k] == 0:
                continue
            spikes_hat_i[k + 1 : k + event_range] *= 0

        event_hat["h{}".format(conv + 1)] = np.where(abs(spikes_hat_i) != 0)[0]

        events.append(event)
        events_hat.append(event_hat)

    return events, events_hat


def miss_false_event(events, events_hat, event_range, conv):
    """
    Helper function to calculate miss and false events
    :param events: true events
    :param events_hat: estimated events
    :param event_range: range of which spike appearance is matched with the true
    :return: (missed_events, missed_per, false_events, false_per)
    """
    num_data = len(events)
    missed_events = np.zeros(num_data)
    false_events = np.zeros(num_data)
    ctr_true_event = 0
    ctr_pred_event = 0
    # loop over all data
    for i in range(num_data):
        event_hn = events[i]["h{}".format(conv + 1)]
        event_hn_hat = events_hat[i]["h{}".format(conv + 1)]
        # loop over true events
        for k in range(len(event_hn)):
            ctr_true_event += 1
            event_distance = event_hn_hat - event_hn[k]
            close_event = event_distance[np.where(abs(event_distance) < event_range)]
            if len(close_event) == 0:
                missed_events[i] += 1
        # loop over predicted events
        for k in range(len(event_hn_hat)):
            ctr_pred_event += 1
            event_distance = event_hn - event_hn_hat[k]
            close_event = event_distance[np.where(abs(event_distance) < event_range)]
            if len(close_event) == 0:
                false_events[i] += 1
    missed_per = 0
    false_per = 0

    if ctr_true_event != 0:
        missed_per = (np.sum(missed_events) / ctr_true_event) * 100
    if ctr_pred_event != 0:
        false_per = (np.sum(false_events) / ctr_pred_event) * 100

    return missed_events, missed_per, false_events, false_per
