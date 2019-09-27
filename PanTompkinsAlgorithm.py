import numpy as np
import hmmlearn.hmm as hmm

from functools import partial
from cardio.models.hmm import HMModel, prepare_hmm_input
from cardio import batchflow as bf
from my_tools import get_annsamples, expand_annotation, get_anntypes, prepare_means_covars, prepare_transmat_startprob



def testPipeline(batch_size=20):
    return (bf.Pipeline()
            .init_variable("qrs_annotation", init_on_each_run=list)
            .load(fmt='wfdb', components=["signal", "meta"])
            .test(dst="qrs_annotation")
            .update_variable("qrs_annotation", bf.B("qrs_annotation"), mode='e')
            .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
            )
def PanTompkinsPipeline(batch_size=20, annot = "pan_tomp_annotation"):
    return (bf.Pipeline()
            .init_variable(annot, init_on_each_run=list)
            .load(fmt='wfdb', components=["signal", "annotation", "meta"], ann_ext='pu1')
            .my_pan_tompkins(dst=annot)
            .update_variable(annot, bf.B(annot), mode='e')
            .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
            )


def low_freq_filter(signal):
    result = np.zeros(len(signal) - 12)
    for index in range(len(result)):
        result[index] = signal[index + 12] - 2 * signal[index + 6] + signal[index]
        if index > 1:
            result[index] += 2 * result[index - 1]
        if index > 2:
            result[index] -= result[index - 2]
    return result


def high_freq_filter(signal):
    result = np.zeros(len(signal) - 32)
    for index in range(len(result)):
        result[index] = 32*signal[index + 16] + 32 * signal[index + 32] - signal[index]
        if index > 1:
            result[index] -= result[index - 1]
    return result


def compute_derivative(signal):
    result = np.zeros(len(signal) - 4)
    for index in range(len(result)):
        result[index] = 0.125 * (2*signal[index + 4] + signal[index + 3] - signal[index + 1] - 2*signal[signal])
    return result
