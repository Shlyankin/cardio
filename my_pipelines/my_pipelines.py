import numpy as np
import hmmlearn.hmm as hmm

from functools import partial
from cardio.models.hmm import HMModel, prepare_hmm_input
from cardio import batchflow as bf
from my_tools import get_annsamples, expand_annotation, get_anntypes, prepare_means_covars, prepare_transmat_startprob


def LoadEcgPipeline(batch_size=20, annot_ext="pu1"):
    """Preprocessing pipeline for Hidden Markov Model.

    This pipeline prepares data for ``hmm_train_pipeline``.
    It works with dataset that generates batches of class ``EcgBatch``.

    Parameters
    ----------
    batch_size : int
        Number of samples in batch.
        Default value is 20.
    features : str
        Batch attribute to store calculated features.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """
    return (bf.Pipeline()
            .load(fmt='wfdb', components=["signal", "annotation", "meta"], ann_ext=annot_ext))

def HMM_preprocessing_pipeline(batch_size=20):
    features = "hmm_features"
    return (bf.Pipeline()
            .init_variable("annsamps", init_on_each_run=list)
            .init_variable("anntypes", init_on_each_run=list)
            .init_variable(features, init_on_each_run=list)
            .cwt(src="signal", dst=features, scales=[4, 8, 16], wavelet="mexh") #применяется прод.вейвлет. преобр.
            .standardize(axis=-1, src=features, dst=features) #преобразуется в посл-ть с единичной дисперсией и c мат.ожиданием 0
            .update_variable("annsamps", bf.F(get_annsamples), mode='e')
            .update_variable("anntypes", bf.F(get_anntypes), mode='e')
            .update_variable(features, bf.B(features), mode='e'))

def HMM_train_pipeline(hmm_preprocessed, batch_size=20, features="hmm_features", channel_ix=0,
                       n_iter=25, random_state=42, model_name='HMM', states=(3, 5, 8, 11, 14, 16)):
    """Train pipeline for Hidden Markov Model.

    This pipeline trains hmm model to isolate QRS, PQ and QT segments.
    It works with dataset that generates batches of class ``EcgBatch``.

    Parameters
    ----------
    hmm_preprocessed : Pipeline
        Pipeline with precomputed hmm features through ``hmm_preprocessing_pipeline``
    batch_size : int
        Number of samples in batch.
        Default value is 20.
    features : str
        Batch attribute to store calculated features.
    channel_ix : int
        Index of signal's channel, which should be used in training and predicting.
    n_iter : int
        Number of learning iterations for ``HMModel``.
    random_state: int
        Random state for ``HMModel``.
    states: list
        States of Markov model.
            0     to states[0] = QRS.
        states[0] to states[1] = ST.
        states[1] to states[2] = T.
        states[2] to states[3] = ISO.to
        states[3] to states[4] = P.
        states[4] to states[5] = PQ.
        Default value is (3, 5, 8, 11, 14, 16).

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """
    lengths = [features_iter.shape[2] for features_iter in hmm_preprocessed.get_variable(features)]
    hmm_features = np.concatenate([features_iter[channel_ix, :, :].T for features_iter
                                   in hmm_preprocessed.get_variable(features)])
    anntype = hmm_preprocessed.get_variable("anntypes")
    annsamp = hmm_preprocessed.get_variable("annsamps")
    expanded = np.concatenate([expand_annotation(samp, types, length) for
                               samp, types, length in zip(annsamp, anntype, lengths)])
    means, covariances = prepare_means_covars(hmm_features, expanded, states=states, num_states=states[5], num_features=3)
    transition_matrix, start_probabilities = prepare_transmat_startprob(states=states)

    config_train = {
        'build': True,
        'estimator': hmm.GaussianHMM(n_components=states[5], n_iter=n_iter, covariance_type="full", random_state=random_state,
                                     init_params='', verbose=False),
        'init_params': {'means_': means, 'covars_': covariances, 'transmat_': transition_matrix,
                        'startprob_': start_probabilities}
    }

    return (bf.Pipeline()
            .init_model("dynamic", HMModel, model_name, config=config_train)
            .load(fmt='wfdb', components=["signal", "annotation", "meta"], ann_ext='pu1')
            .cwt(src="signal", dst=features, scales=[4, 8, 16], wavelet="mexh")
            .standardize(axis=-1, src=features, dst=features)
            .train_model(model_name, make_data=partial(prepare_hmm_input, features=features, channel_ix=channel_ix)))

def HMM_predict_pipeline(model_path, batch_size=20, features="hmm_features",
                         channel_ix=0, annot="hmm_annotation", model_name='HMM'):
    """Prediction pipeline for Hidden Markov Model.

    This pipeline isolates QRS, PQ and QT segments.
    It works with dataset that generates batches of class ``EcgBatch``.

    Parameters
    ----------
    model_path : str
        Path to pretrained ``HMModel``.
    batch_size : int
        Number of samples in batch.
        Default value is 20.
    features : str
        Batch attribute to store calculated features.
    channel_ix : int
        Index of channel, which data should be used in training and predicting.
    annot: str
        Specifies attribute of batch in which annotation will be stored.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """

    config_predict = {
        'build': False,
        'load': {'path': model_path}
    }

    return (bf.Pipeline()
            .init_model("static", HMModel, model_name, config=config_predict)
            .cwt(src="signal", dst=features, scales=[4, 8, 16], wavelet="mexh")
            .standardize(axis=-1, src=features, dst=features)
            .predict_model(model_name, make_data=partial(prepare_hmm_input, features=features, channel_ix=channel_ix),
                           save_to=bf.B(annot), mode='w')
            .calc_ecg_parameters(src=annot))


def PanTompkinsPipeline(batch_size=20, annot = "pan_tomp_annotation"):
    return (bf.Pipeline()
            .init_variable(annot, init_on_each_run=list)
            .my_pan_tompkins(dst=annot)
            .update_variable(annot, bf.B(annot), mode='e'))
