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
            .my_pan_tompkins(dst=annot)
            .update_variable(annot, bf.B(annot), mode='e')
            .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
            )

def HilbertTransformPipeline(batch_size=20, annot = "hilbert_annotation"):
    return (bf.Pipeline()
            .init_variable(annot, init_on_each_run=list)
            .load(fmt='wfdb', components=["signal", "meta"])
            .band_pass_signals(8, 20)
            .hilbert_transform(dst=annot)
            .update_variable(annot, bf.B(annot), mode='e')
            .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True))
