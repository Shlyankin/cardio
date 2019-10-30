import numpy as np
import matplotlib.pyplot as plt
from cardio import EcgBatch


def calculate_metrics(batch, states, state_num, annot):
    parameters = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for i in range(len(batch.annotation)):
        # prepare annotations
        anntype = batch.annotation[i]["anntype"]
        annsamp = batch.annotation[i]["annsamp"]
        expanded = expand_annotation(annsamp, anntype, len(batch.signal[0][0]))
        # calculate parameters for each record and sum them
        new_parameters = tp_tn_fp_fn(expanded, batch.get(component=annot)[i], states, state_num)
        parameters["tp"] += new_parameters["tp"]
        parameters["fn"] += new_parameters["fn"]
        parameters["fp"] += new_parameters["fp"]
        parameters["tn"] += new_parameters["tn"]
    accuracy = (parameters["tp"] + parameters["tn"]) / (
                parameters["tp"] + parameters["tn"] + parameters["fp"] + parameters["fn"])
    precision = (parameters["tp"]) / (parameters["tp"] + parameters["fp"])
    recall = (parameters["tp"]) / (parameters["tp"] + parameters["fn"])
    fscore = 2 * precision * recall / (precision + recall)
    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f-score": fscore,
            "tp": parameters["tp"],
            "fn": parameters["fn"],
            "fp": parameters["fp"],
            "tn": parameters["tn"]}


def tp_tn_fp_fn(true_annot, annot, states, state_num):
    def prepare_annot(hmm_annotation, inter_val):
        intervals = np.zeros(hmm_annotation.shape, dtype=np.int8)
        for val in inter_val:
            intervals = np.logical_or(intervals, (hmm_annotation == val).astype(np.int8)).astype(np.int8)
        return intervals

    # TODO: check range
    range1 = np.array(list(range(state_num, state_num + 1)), np.int64)
    range2 = np.array(list(range(states[state_num - 1] if state_num != 0 else 0, states[state_num])), np.int64)
    prepared_true_annot = prepare_annot(true_annot, range1)
    prepared_annot = prepare_annot(annot, range2)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(prepared_annot)):
        if prepared_true_annot[i] == 1 and prepared_annot[i] == 1:
            tp += 1
        if prepared_true_annot[i] == 0 and prepared_annot[i] == 0:
            tn += 1
        if prepared_true_annot[i] == 1 and prepared_annot[i] == 0:
            fn += 1
        if prepared_true_annot[i] == 0 and prepared_annot[i] == 1:
            fp += 1
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def calculate_old_metrics(batch, states, state_num, annot):
    def tp_fn_fp(true_annot, annot, states, state_num, error):
        true_intervals = find_intervals_borders(true_annot, np.array(list(range(state_num if state_num != 0 else 0,
                                                                                state_num + 1)), np.int64))
        intervals = find_intervals_borders(annot, np.array(list(range(states[state_num - 1] if state_num != 0 else 0,
                                                                      states[state_num])), np.int64))
        tp = 0
        for i in range(len(true_intervals[0])):
            for j in range(len(intervals[0])):
                if abs(true_intervals[0][i] - intervals[0][j]) < error and abs(
                        true_intervals[1][i] - intervals[1][j]) < error:
                    tp += 1
                    break
        fn = len(true_intervals[0]) - tp
        fp = len(intervals[0]) - tp
        return {"tp": tp, "fn": fn, "fp": fp}

    # ------------ body of function ------------
    parameters = {"tp": 0, "fn": 0, "fp": 0}
    for i in range(len(batch.annotation)):
        # 100ms is max different between experts annotations and given annotation
        error = batch.meta[i]["fs"] * 0.1
        anntype = batch.annotation[i]["anntype"]
        annsamp = batch.annotation[i]["annsamp"]
        expanded = expand_annotation(annsamp, anntype, len(batch.signal[0][0]))
        new_parameters = tp_fn_fp(expanded, batch.get(component=annot)[i], states, state_num, error=error)
        parameters["tp"] += new_parameters["tp"]
        parameters["fn"] += new_parameters["fn"]
        parameters["fp"] += new_parameters["fp"]
    return {"sensitivity": float(parameters["tp"]) / (parameters["tp"] + parameters["fn"]),
            "specificity": float(parameters["tp"]) / (parameters["tp"] + parameters["fp"]),
            "tp": parameters["tp"],
            "fn": parameters["fn"],
            "fp": parameters["fp"]}


def find_intervals_borders(hmm_annotation, inter_val):
    intervals = np.zeros(hmm_annotation.shape, dtype=np.int8)
    for val in inter_val:
        intervals = np.logical_or(intervals, (hmm_annotation == val).astype(np.int8)).astype(np.int8)
    masque = np.diff(intervals)
    starts = np.where(masque == 1)[0] + 1
    ends = np.where(masque == -1)[0] + 1
    if np.any(inter_val == hmm_annotation[:1]):
        ends = ends[1:]
    if np.any(inter_val == hmm_annotation[-1:]):
        starts = starts[:-1]
    return starts, ends


def prepare_means_covars(hmm_features, clustering, states=(3, 5, 11, 14, 17, 19), num_states=19, num_features=3):
    """This function is specific to the task and the model configuration, thus contains hardcode.
    """
    means = np.zeros((num_states, num_features))
    covariances = np.zeros((num_states, num_features, num_features))

    # Prepearing means and variances
    last_state = 0
    unique_clusters = len(np.unique(clustering)) - 1  # Excuding value -1, which represents undefined state
    for state, cluster in zip(states, np.arange(unique_clusters)):
        value = hmm_features[clustering == cluster, :]
        means[last_state:state, :] = np.mean(value, axis=0)
        covariances[last_state:state, :, :] = value.T.dot(value) / np.sum(clustering == cluster)
        last_state = state

    return means, covariances


def prepare_transmat_startprob(states=(3, 5, 11, 14, 17, 19)):
    """
        Function calculate transition matrix (матрицу переходов) and start probabilities (вектор начальных вероятностей)
    """
    # Transition matrix - each row should add up tp 1
    transition_matrix = np.diag(states[5] * [14 / 15.0]) + np.diagflat((states[5] - 1) * [1 / 15.0], 1) + np.diagflat(
        [1 / 15.0], -(states[5] - 1))

    # We suppose that absence of P-peaks is possible
    # P-peaks may not appear
    transition_matrix[states[3] - 1, states[3]] = 0.9 * 1 / 15.0
    transition_matrix[states[3] - 1, states[4]] = 0.1 * 1 / 15.0

    # Initial distribution - should add up to 1
    start_probabilities = np.array(states[5] * [1 / np.float(states[5])])

    return transition_matrix, start_probabilities


def expand_annotation(annsamp, anntype, length):
    """
        Unravel annotation
    """
    begin = -1
    end = -1
    s = 'none'
    states = {'N': 0, 'st': 1, 't': 2, 'iso': 3, 'p': 4, 'pq': 5}
    annot_expand = -1 * np.ones(length)

    for j, samp in enumerate(annsamp):
        if anntype[j] == '(':
            begin = samp
            if (end > 0) & (s != 'none'):
                if s == 'N':
                    annot_expand[end:begin] = states['st']
                elif s == 't':
                    annot_expand[end:begin] = states['iso']
                elif s == 'p':
                    annot_expand[end:begin] = states['pq']
        elif anntype[j] == ')':
            end = samp
            if (begin > 0) & (s != 'none'):
                annot_expand[begin:end] = states[s]
        else:
            s = anntype[j]

    return annot_expand


def get_annsamples(batch):
    """Get annsamples from annotation
    """
    return [ann["annsamp"] for ann in batch.annotation]


def get_anntypes(batch):
    """Get anntypes from annotation
    """
    return [ann["anntype"] for ann in batch.annotation]
