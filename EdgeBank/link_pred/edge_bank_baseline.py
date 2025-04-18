"""
EdgeBank
"""
from pathlib import Path

import numpy
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import random
import time
from sklearn.metrics import *
from tqdm import tqdm
import math
from collections import Counter

from edge_sampler import RandEdgeSampler, RandEdgeSampler_adversarial
from load_data import Data, get_data
from args_parser import parse_args_edge_bank
from evaluation import *

np.random.seed(0)
random.seed(0)


def predict_links(memory, edge_set):
    """
    Predict whether each edge in edge_set is an actual or a dummy edge based on the memory
    """
    source_nodes, destination_nodes = edge_set
    # prediction of an edge
    pred = []
    for i in range(len(destination_nodes)):
        if (source_nodes[i], destination_nodes[i]) in memory:
            pred.append(1)
        else:
            pred.append(0)

    return np.array(pred)


def edge_bank_unlimited_memory(sources_list, destinations_list):
    """
    generates the memory of EdgeBank
    The memory stores every edges that it has seen
    """
    # generate memory
    mem_edges = {}
    for e_idx in range(len(sources_list)):
        if (sources_list[e_idx], destinations_list[e_idx]) not in mem_edges:
            mem_edges[(sources_list[e_idx], destinations_list[e_idx])] = 1

    return mem_edges


def edge_bank_repetition_based_memory(sources_list, destinations_list):
    """
    in memory, save edges that has repeated more than a threshold
    """
    # frequency of seeing each edge
    all_seen_edges = {}
    for e_idx in range(len(sources_list)):
        if (sources_list[e_idx], destinations_list[e_idx]) in all_seen_edges:
            all_seen_edges[(sources_list[e_idx], destinations_list[e_idx])] += 1
        else:
            all_seen_edges[(sources_list[e_idx], destinations_list[e_idx])] = 1
    n_repeat = np.array(list(all_seen_edges.values()))


    threshold = np.mean(n_repeat)

    mem_edges = {}
    for edge, n_e_repeat in all_seen_edges.items():
        if n_e_repeat >= threshold:
            mem_edges[edge] = 1


    return mem_edges


def time_window_edge_memory(sources_list, destinations_list, timestamps_list, start_time, end_time):
    """
    returns a memory that contains all edges seen during a time window
    """
    mem_mask = np.logical_and(timestamps_list <= end_time, timestamps_list >= start_time)
    src_in_window = sources_list[mem_mask]
    dst_in_window = destinations_list[mem_mask]
    mem_edges = edge_bank_unlimited_memory(src_in_window, dst_in_window)
    return mem_edges



def edge_bank_time_window_memory(sources_list, destinations_list, timestamps_list, window_mode, memory_span=0.15):
    """
    only saves the edges seen the time time interval equal to the last time window in timestamps_list
    """
    # print("Info: Total number of edges:", len(sources_list))
    if window_mode == 'fixed':
        window_start_ts = np.quantile(timestamps_list, 1 - memory_span)
        window_end_ts = max(timestamps_list)
    elif window_mode == 'avg_reoccur':
        e_ts_l = {}
        for e_idx in range(len(sources_list)):
            curr_edge = (sources_list[e_idx], destinations_list[e_idx])
            if curr_edge not in e_ts_l:
                e_ts_l[curr_edge] = []
            e_ts_l[curr_edge].append(timestamps_list[e_idx])

        sum_t_interval = 0
        for e, ts_list in e_ts_l.items():
            if len(ts_list) > 1:
                ts_interval_l = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
                sum_t_interval += np.mean(ts_interval_l)
        avg_t_interval = sum_t_interval / len(e_ts_l)
        window_end_ts = max(timestamps_list)
        window_start_ts = window_end_ts - avg_t_interval

    # print("Info: Time window mode:", window_mode)
    # print(f"Info: start window: {window_start_ts}, end window: {window_end_ts}, "
    #       f"interval: {window_end_ts - window_start_ts}")
    mem_edges = time_window_edge_memory(sources_list, destinations_list, timestamps_list, start_time=window_start_ts,
                                        end_time=window_end_ts)
    # print("Info: EdgeBank memory mode: >> Time Window Memory <<")
    # print(f"Info: Memory contains {len(mem_edges)} edges.")

    return mem_edges


# New addition: Instead of predicting a link based on if the link exists in memory, 
# we sample based on the frequency the edge has existed in the past
def edge_bank_collect_frequencies(sources_list, destinations_list):
    """
    collect number of times each edge (u,v) has been seen in the past
    """
    freq_dict = {}
    for e_idx in range(len(sources_list)):
        edge = (sources_list[e_idx], destinations_list[e_idx])
        if edge not in freq_dict:
            freq_dict[edge] = 1
        else:
            freq_dict[edge] += 1
    return freq_dict

def predict_links_frequency(freq_memory, edge_set):
    """
    Returns a memory dictionary where the score for each edge is proportional
    to the log-scaled frequency of its appearance in the training set.
    """
    source_nodes, destination_nodes = edge_set
    preds = []
    for i in range(len(source_nodes)):
        edge = (source_nodes[i], destination_nodes[i])
        # preds.append(freq_memory.get(edge, 0))
        preds.append(math.log1p(freq_memory.get(edge, 0)))
        # #or 
        # max_freq = max(freq_memory.values()) if len(freq_memory) > 0 else 1
        # score = freq_memory.get(edge, 0) / max_freq
        # preds.append(score)


    return np.array(preds)


# Now lets do a window based version of this
def edge_bank_collect_frequencies_in_window(sources_list, destinations_list, timestamps_list, start_time, end_time):
    """
    Collect number of times each edge (u,v) has been seen in the past within a specified time window.
    """
    mem_mask = np.logical_and(timestamps_list <= end_time, timestamps_list >= start_time)
    src_in_window = sources_list[mem_mask]
    dst_in_window = destinations_list[mem_mask]

    freq_dict = {}
    for i in range(len(src_in_window)):
        edge = (src_in_window[i], dst_in_window[i])
        freq_dict[edge] = freq_dict.get(edge, 0) + 1
    return freq_dict


def edge_bank_window_frequency_memory(sources_list, destinations_list, timestamps_list, window_mode, memory_span=0.15):
    """
    Builds a frequency-based memory using only the edges seen in a given time window.
    The value for each edge is its frequency (or log-scaled frequency) during that window.
    """

    if window_mode == 'fixed':
        window_start_ts = np.quantile(timestamps_list, 1 - memory_span)
        window_end_ts = max(timestamps_list)

    elif window_mode == 'avg_reoccur':
        e_ts_l = {}
        for e_idx in range(len(sources_list)):
            curr_edge = (sources_list[e_idx], destinations_list[e_idx])
            if curr_edge not in e_ts_l:
                e_ts_l[curr_edge] = []
            e_ts_l[curr_edge].append(timestamps_list[e_idx])

        sum_t_interval = 0
        for ts_list in e_ts_l.values():
            if len(ts_list) > 1:
                ts_interval_l = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
                sum_t_interval += np.mean(ts_interval_l)
        avg_t_interval = sum_t_interval / len(e_ts_l)
        window_end_ts = max(timestamps_list)
        window_start_ts = window_end_ts - avg_t_interval

    freq_memory = edge_bank_collect_frequencies_in_window(
        sources_list, destinations_list, timestamps_list,
        start_time=window_start_ts, end_time=window_end_ts
    )

    return freq_memory


def edge_bank_link_pred_end_to_end(history_data, positive_edges, negative_edges, memory_opt):
    """
    EdgeBank link prediction
    """
    srcs = history_data.sources
    dsts = history_data.destinations
    ts_list = history_data.timestamps
    pos_sources, pos_destinations = positive_edges
    neg_sources, neg_destinations = negative_edges
    assert (len(srcs) == len(dsts))
    assert (len(pos_sources) == len(pos_destinations))
    assert (len(neg_sources) == len(neg_destinations))

    if memory_opt['m_mode'] == 'unlim_mem':
        mem_edges = edge_bank_unlimited_memory(srcs, dsts)
        predict_fn = predict_links
    elif memory_opt['m_mode'] == 'repeat_freq':
        mem_edges = edge_bank_repetition_based_memory(srcs, dsts)
        predict_fn = predict_links
    elif memory_opt['m_mode'] == 'time_window':
        mem_edges = edge_bank_time_window_memory(srcs, dsts, ts_list, memory_opt['w_mode'])
        predict_fn = predict_links
    # new frequency based memory weight mode
    elif memory_opt['m_mode'] == 'freq_weight':
        mem_edges = edge_bank_collect_frequencies(srcs, dsts)
        predict_fn = predict_links_frequency
    # new frequency based time window memory weight mode
    elif memory_opt['m_mode'] == 'window_freq_weight':
        mem_edges = edge_bank_window_frequency_memory(srcs, dsts, ts_list, memory_opt['w_mode'])
        predict_fn = predict_links_frequency

    else:
        mem_edges = {}
        print("Undefined Memory Option!")
        exit(-1)

    pos_pred = predict_fn(mem_edges, positive_edges)
    neg_pred = predict_fn(mem_edges, negative_edges)

    return pos_pred, neg_pred


def edge_bank_link_pred_batch(train_val_data, test_data, rand_sampler, args):
    """
    EdgeBank link prediction: batch based
    """
    assert rand_sampler.seed is not None
    rand_sampler.reset_random_state()

    TEST_BATCH_SIZE = args['batch_size']
    num_test_instance = len(test_data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    agg_pred_score, agg_true_label = [], []
    val_ap, val_auc_roc, measures_list = [], [], []

    # for k in tqdm(range(num_test_batch)):
    for k in range(num_test_batch):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
        sources_batch = test_data.sources[s_idx:e_idx]
        destinations_batch = test_data.destinations[s_idx:e_idx]
        timestamps_batch = test_data.timestamps[s_idx:e_idx]
        # edge_idxs_batch = test_data.edge_idxs[s_idx: e_idx]
        positive_edges = (sources_batch, destinations_batch)

        size = len(sources_batch)  # number of negative edges
        if rand_sampler.neg_sample != 'rnd':
            src_negative_samples, dst_negative_samples = rand_sampler.sample(size, sources_batch, destinations_batch,
                                                                             timestamps_batch[0],
                                                                             timestamps_batch[-1])
        else:
            src_negative_samples, dst_negative_samples = rand_sampler.sample(size, sources_batch, destinations_batch)
            src_negative_samples = sources_batch  # similar to what baselines do

        negative_edges = (src_negative_samples, dst_negative_samples)

        pos_label = np.ones(size)
        neg_label = np.zeros(size)
        true_label = np.concatenate([pos_label, neg_label])
        agg_true_label = np.concatenate([agg_true_label, true_label])

        if args['learn_through_time']:
            history_data = Data(np.concatenate([train_val_data.sources, test_data.sources[: s_idx]]),
                                np.concatenate([train_val_data.destinations, test_data.destinations[: s_idx]]),
                                np.concatenate([train_val_data.timestamps, test_data.timestamps[: s_idx]]),
                                np.concatenate([train_val_data.edge_idxs, test_data.edge_idxs[: s_idx]]),
                                np.concatenate([train_val_data.labels, test_data.labels[: s_idx]]))
        else:
            history_data = train_val_data

        # performance evaluation
        pos_pred, neg_pred = edge_bank_link_pred_end_to_end(history_data, positive_edges, negative_edges, args)
        pred_score = np.concatenate([pos_pred, neg_pred])
        agg_pred_score = np.concatenate([agg_pred_score, pred_score])
        assert (len(pred_score) == len(true_label)), "Lengths of predictions and true labels do not match!"

        val_ap.append(average_precision_score(true_label, pred_score))
        val_auc_roc.append(roc_auc_score(true_label, pred_score))

        # extra performance measures
        measures_dict = extra_measures(true_label, pred_score)
        measures_list.append(measures_dict)
    measures_df = pd.DataFrame(measures_list)
    avg_measures_dict = measures_df.mean()

    return np.mean(val_ap), np.mean(val_auc_roc), avg_measures_dict


def main():
    """
    EdgeBank main execution procedure
    """
    print("===========================================================================")
    cm_args = parse_args_edge_bank()
    print("===========================================================================")
    # arguments
    network_name = cm_args.data
    val_ratio = cm_args.val_ratio
    test_ratio = cm_args.test_ratio
    n_runs = cm_args.n_runs
    NEG_SAMPLE = cm_args.neg_sample
    learn_through_time = True  # similar to memory of TGN
    args = {'network_name': network_name,
            'n_runs': n_runs,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'm_mode': cm_args.mem_mode,
            'w_mode': cm_args.w_mode,
            'learn_through_time': learn_through_time,
            'batch_size': 200,
            'neg_sample': NEG_SAMPLE}

    # path
    common_path = f'{Path(__file__).parents[1]}/data/data/'
    # ebank_log_file = "{}/ebank_logs/EdgeBank_{}_self_sup.log".format(common_path, network_name)

    # load data
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_data(common_path, network_name, val_ratio, test_ratio)

    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    tr_val_data = Data(np.concatenate([train_data.sources, val_data.sources]),
                       np.concatenate([train_data.destinations, val_data.destinations]),
                       np.concatenate([train_data.timestamps, val_data.timestamps]),
                       np.concatenate([train_data.edge_idxs, val_data.edge_idxs]),
                       np.concatenate([train_data.labels, val_data.labels]))

    # define negative edge sampler
    if NEG_SAMPLE != 'rnd':
        print("INFO: Negative Edge Sampling: {}".format(NEG_SAMPLE))
        test_rand_sampler = RandEdgeSampler_adversarial(full_data.sources, full_data.destinations, full_data.timestamps,
                                                        val_data.timestamps[-1], NEG_SAMPLE, seed=2)
    else:
        test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)


    # executing different runs
    for i_run in range(n_runs):
        print("INFO:root:****************************************")
        for k, v in args.items():
            print("INFO:root:{}: {}".format(k, v))
        print ("INFO:root:Run: {}".format(i_run))
        start_time_run = time.time()
        inherent_ap, inherent_auc_roc, avg_measures_dict = edge_bank_link_pred_batch(tr_val_data,
                                                                                     test_data, test_rand_sampler,
                                                                                     args)
        print('INFO:root:Test statistics: Old nodes -- auc_inherent: {}'.format(inherent_auc_roc))
        print('INFO:root:Test statistics: Old nodes -- ap_inherent: {}'.format(inherent_ap))
        # extra performance measures
        # Note: just prints out for the Test set! in transductive setting
        for measure_name, measure_value in avg_measures_dict.items():
            print ('INFO:root:Test statistics: Old nodes -- {}: {}'.format(measure_name, measure_value))

        elapse_time = time.time() - start_time_run
        print('INFO:root:EdgeBank: Run: {}, Elapsed time: {}'.format(i_run, elapse_time))
        print('INFO:root:****************************************')

    print("===========================================================================")


if __name__ == '__main__':
    main()
