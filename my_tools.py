import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from cardio import EcgBatch

def calc_metr_qrs(batch, annot, type='micro'):
    def transform(annotation):
        indexes = annotation == 0
        annotation[annotation != 0] = 0
        annotation[indexes] = 1

    """    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']"""
    metr = {"accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f-score": 0}
    seq_pred = np.array([], dtype=np.int32)
    seq_true = np.array([], dtype=np.int32)
    for i in range(len(batch.annotation)):
        anntype = batch.annotation[i]["anntype"]
        annsamp = batch.annotation[i]["annsamp"]
        expanded = expand_annotation(annsamp, anntype, len(batch.signal[i][0]))
        expanded = expanded.astype(np.int32)
        transform(expanded)
        seq_true = np.concatenate((seq_true, expanded))
        annotation = batch.get(component=annot)[i]
        seq_pred = np.concatenate((seq_pred, annotation))
    metr["precision"] = metrics.precision_score(y_pred=seq_pred, y_true=seq_true, average=type)
    metr["recall"] = metrics.recall_score(y_pred=seq_pred, y_true=seq_true, average=type)
    metr["f-score"] = metrics.f1_score(y_pred=seq_pred, y_true=seq_true, average=type)
    metr["accuracy"] = metrics.accuracy_score(y_pred=seq_pred, y_true=seq_true)
    return metr

def calc_metr(seq_pred, seq_true, type='micro'):
    metr = {"accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f-score": 0}
    metr["precision"] = metrics.precision_score(y_pred=seq_pred, y_true=seq_true, labels=[0, 1, 2, 3, 4, 5], average=type)
    metr["recall"] = metrics.recall_score(y_pred=seq_pred, y_true=seq_true, labels=[0, 1, 2, 3, 4, 5], average=type)
    metr["f-score"] = metrics.f1_score(y_pred=seq_pred, y_true=seq_true, labels=[0, 1, 2, 3, 4, 5], average=type)
    metr["accuracy"] = metrics.accuracy_score(y_pred=seq_pred, y_true=seq_true)
    return metr



def calc_metr_batch(batch, annot, states, type='micro'):
    """    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']"""
    metr = {"accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f-score": 0}
    seq_pred = np.array([], dtype=np.int32)
    seq_true = np.array([], dtype=np.int32)
    for i in range(len(batch.annotation)):
        anntype = batch.annotation[i]["anntype"]
        annsamp = batch.annotation[i]["annsamp"]
        expanded = expand_annotation(annsamp, anntype, len(batch.signal[i][0]))
        expanded[expanded == -1] = 0
        expanded = expanded.astype(np.int32)
        seq_true = np.concatenate((seq_true, expanded))
        transformed_annotation = transform_annot(batch.get(component=annot)[i], states)
        seq_pred = np.concatenate((seq_pred, transformed_annotation))
    metr["precision"] = metrics.precision_score(y_pred=seq_pred, y_true=seq_true, labels=[0, 1, 2, 3, 4, 5], average=type)
    metr["recall"] = metrics.recall_score(y_pred=seq_pred, y_true=seq_true, labels=[0, 1, 2, 3, 4, 5], average=type)
    metr["f-score"] = metrics.f1_score(y_pred=seq_pred, y_true=seq_true, labels=[0, 1, 2, 3, 4, 5], average=type)
    metr["accuracy"] = metrics.accuracy_score(y_pred=seq_pred, y_true=seq_true)
    return metr

def calcuate_metrics_for_all_model(batch, annot, model_states):
    def tp_tn_fp_fn(true_annot, annot, model_states):
        states = {0: "N", 1: "ST", 2: "T", 3: "ISO", 4: "P", 5: "PQ"}
        metrics = {
            "N":    {"t": 0, "f": 0},
            "ST":   {"t": 0, "f": 0},
            "T":    {"t": 0, "f": 0},
            "ISO":  {"t": 0, "f": 0},
            "P":    {"t": 0, "f": 0},
            "PQ":   {"t": 0, "f": 0},
        }
        temp_annot = np.ndarray(len(annot))
        for i in range(len(annot)):
            if annot[i] in range(0, model_states[0]): temp_annot[i] = 0
            if annot[i] in range(model_states[0], model_states[1]): temp_annot[i] = 1
            if annot[i] in range(model_states[1], model_states[2]): temp_annot[i] = 2
            if annot[i] in range(model_states[2], model_states[3]): temp_annot[i] = 3
            if annot[i] in range(model_states[3], model_states[4]): temp_annot[i] = 4
            if annot[i] in range(model_states[4], model_states[5]): temp_annot[i] = 5
        for i in range(len(temp_annot)):
            if true_annot[i] == -1: continue
            if true_annot[i] == temp_annot[i]:
                metrics[states[true_annot[i]]]["t"] += 1
            else:
                metrics[states[true_annot[i]]]["f"] += 1
        return metrics

    states = {0: "N", 1: "ST", 2: "T", 3: "ISO", 4: "P", 5: "PQ"}
    states_parameters = {
            "N":    {"t": 0, "f": 0},
            "ST":   {"t": 0, "f": 0},
            "T":    {"t": 0, "f": 0},
            "ISO":  {"t": 0, "f": 0},
            "P":    {"t": 0, "f": 0},
            "PQ":   {"t": 0, "f": 0},
    }
    parameters = {"t": 0, "f": 0}
    for i in range(len(batch.annotation)):
        # prepare annotations
        anntype = batch.annotation[i]["anntype"]
        annsamp = batch.annotation[i]["annsamp"]
        expanded = expand_annotation(annsamp, anntype, len(batch.signal[0][0]))
        # calculate parameters for each record and sum them
        new_parameters = tp_tn_fp_fn(expanded, batch.get(component=annot)[i], model_states)
        for state in states.values():
            states_parameters[state]["t"] += new_parameters[state]["t"]
            states_parameters[state]["f"] += new_parameters[state]["f"]
    for state in states.values():
        parameters["t"] += states_parameters[state]["t"]
        parameters["f"] += states_parameters[state]["f"]
    accuracy = parameters["t"] / (parameters["f"] + parameters["t"])
    return {"states_parameters": states_parameters,
            "accuracy": accuracy}


def calculate_metrics(batch, states, state_num, annot):
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


def calculate_old_metrics(batch, annot, states, state_num):
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

# states = {'N': 0, 'st': 1, 't': 2, 'iso': 3, 'p': 4, 'pq': 5}
def transform_annot(annot, states):
    states = np.insert(states, 0, 0)
    for state in range(1, 7):
        annot[annot < states[state]] = states[-1] + state - 1
    for state in range(6):
        annot[annot == states[-1] + state] = state
    return annot

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
