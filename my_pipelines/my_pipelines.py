import numpy as np
import hmmlearn.hmm as hmm

from functools import partial

from cardio.batchflow import V, F
from cardio.batchflow.batchflow.models import tf
from cardio.models import DirichletModel, concatenate_ecg_batch
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

def HilbertTransformPipeline(batch_size=20, annot = "hilbert_annotation"):
    return (bf.Pipeline()
            .init_variable(annot, init_on_each_run=list)
            .load(fmt='wfdb', components=["signal", "meta"])
            .band_pass_signals(8, 20)
            .hilbert_transform(dst=annot)
            .update_variable(annot, bf.B(annot), mode='e'))

def PanTompkinsPipeline(batch_size=20, annot = "pan_tomp_annotation"):
    return (bf.Pipeline()
            .init_variable(annot, init_on_each_run=list)
            .my_pan_tompkins(dst=annot)
            .update_variable(annot, bf.B(annot), mode='e'))

def dirichlet_train_pipeline(labels_path, batch_size=256, n_epochs=1000, gpu_options=None,
                             loss_history='loss_history', model_name='dirichlet'):
    """Train pipeline for Dirichlet model.

    This pipeline trains Dirichlet model to find propability of atrial fibrillation.
    It works with dataset that generates batches of class ``EcgBatch``.

    Parameters
    ----------
    labels_path : str
        Path to csv file with true labels.
    batch_size : int
        Number of samples per gradient update.
        Default value is 256.
    n_epochs : int
        Number of times to iterate over the training data arrays.
        Default value is 1000.
    gpu_options : GPUOptions
        An argument for tf.ConfigProto ``gpu_options`` proto field.
        Default value is ``None``.
    loss_history : str
        Name of pipeline variable to save loss values to.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """

    model_config = {
        "session": {"config": tf.ConfigProto(gpu_options=gpu_options)},
        "input_shape": F(lambda batch: batch.signal[0].shape[1:]),
        "class_names": F(lambda batch: batch.label_binarizer.classes_),
        "loss": None,
    }

    return (bf.Pipeline()
            .init_model("dynamic", DirichletModel, name=model_name, config=model_config)
            .init_variable(loss_history, init_on_each_run=list)
            .load(components=["signal", "meta"], fmt="wfdb")
            .load(components="target", fmt="csv", src=labels_path)
            .drop_labels(["~"])
            .rename_labels({"N": "NO", "O": "NO"})
            .flip_signals()
            .random_resample_signals("normal", loc=300, scale=10)
            .random_split_signals(2048, {"A": 9, "NO": 3})
            .binarize_labels()
            .train_model(model_name, make_data=concatenate_ecg_batch,
                         fetches="loss", save_to=V(loss_history), mode="a")
            .run(batch_size=batch_size, shuffle=True, drop_last=True, n_epochs=n_epochs, lazy=True))