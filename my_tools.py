import numpy as np
import matplotlib.pyplot as plt
from cardio import EcgBatch


def calculate_sensitivity(batch):
    anntypes = batch.annotation["anntype"]
    annsamps = batch.annotation["annsamp"]
    for annsamp, anntype, hmm_annotation in annsamps, anntypes, batch.hmm_annotation:
        expanded = expand_annotation(annsamp, anntype, len(batch.signal[0][0]))
        parameters = tp_fn_fp_count(true_annot=expanded, annot=hmm_annotation)
    return float(parameters["tp"]) / (parameters["tp"] + parameters["fn"])

def tp_fn_fp_count(true_annot, annot):
    return {"tp" : 100, "fn" : 100, "fp" : 100}

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
    """ This function is specific to the task and the model configuration, thus contains hardcode.
    """
    # Transition matrix - each row should add up tp 1
    transition_matrix = np.diag(states[5] * [14 / 15.0]) + np.diagflat((states[5]-1) * [1 / 15.0], 1) + np.diagflat([1 / 15.0], -(states[5]-1))

    # We suppose that absence of P-peaks is possible
    transition_matrix[states[3]-1, states[3]] = 0.9 * 1 / 15.0
    transition_matrix[states[3]-1, states[4]] = 0.1 * 1 / 15.0

    # Initial distribution - should add up to 1
    start_probabilities = np.array(states[5] * [1 / np.float(states[5])])

    return transition_matrix, start_probabilities


def expand_annotation(annsamp, anntype, length):
    """Unravel annotation
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
