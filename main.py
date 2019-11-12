import os
import numpy as np
import cardio.batchflow as bf
from cardio import EcgBatch
from my_pipelines.my_pipelines import PanTompkinsPipeline, HMM_predict_pipeline, LoadEcgPipeline, HilbertTransformPipeline
from my_tools import calculate_old_metrics, calculate_metrics
import warnings

warnings.filterwarnings('ignore')

def print_old_metrics(batch, model_name, states, type_states):
    parameters = {"tp": 0, "fn": 0, "fp": 0}
    for type_state in type_states:
        print(type_states[type_state])
        states = all_states[model_name]
        state_parameters = calculate_old_metrics(batch, np.array(list(states), np.int64), type_state,
                                           "hmm_annotation" + model_name)
        print(model_name + " \tsensitivity= " + str(state_parameters["sensitivity"]) + "\tspecificity= " + str(
            state_parameters["specificity"]))
        parameters["tp"] += state_parameters["tp"]
        parameters["fn"] += state_parameters["fn"]
        parameters["fp"] += state_parameters["fp"]
    sensitivity = float(parameters["tp"]) / (parameters["tp"] + parameters["fn"])
    specificity = float(parameters["tp"]) / (parameters["tp"] + parameters["fp"])
    print(model_name + " \tsensitivity= " + str(sensitivity) + "\tspecificity= " + str(specificity))

def print_metrics(batch, model_name, states, type_states):
    parameters = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for type_state in type_states:
        print(type_states[type_state], end='\t')
        states = all_states[model_name]
        state_parameters = calculate_metrics(batch, np.array(list(states), np.int64), type_state, "hmm_annotation" + model_name)
        for i in parameters.keys():
            parameters[i] += state_parameters[i]
        print(model_name +
              "\taccuracy= " + str(state_parameters["accuracy"]) +
              "\tprecision= " + str(state_parameters["precision"]) +
              "\trecall= " + str(state_parameters["recall"]) +
              "\tf-score= " + str(state_parameters["f-score"]))
    accuracy = (parameters["tp"] + parameters["tn"]) / (
            parameters["tp"] + parameters["tn"] + parameters["fp"] + parameters["fn"])
    precision = (parameters["tp"]) / (parameters["tp"] + parameters["fp"])
    recall = (parameters["tp"]) / (parameters["tp"] + parameters["fn"])
    fscore = 2 * precision * recall / (precision + recall)
    print(model_name +
          "\taccuracy= "  + str(accuracy)   +
          "\tprecision= " + str(precision)  +
          "\trecall= "    + str(recall)     +
          "\tf-score= "   + str(fscore))

"""
sys.path.append("..")
index = bf.FilesIndex(path="data/A*.hea", no_ext=True, sort=True)
print(index.indices)
eds = bf.Dataset(index, batch_class=EcgBatch)
batch = eds.next_batch(batch_size=2)
batch_with_data = batch.load(fmt="wfdb", components=["signal", "meta"])
batch_with_data.show_ecg('A00001', start=10, end=15)
"""

SIGNALS_PATH = "data\\qt-database-1.0.0"  # set path to QT database
SIGNALS_MASK = os.path.join(SIGNALS_PATH, "*.hea")

index = bf.FilesIndex(path=SIGNALS_MASK, no_ext=True, sort=True)
dtst = bf.Dataset(index, batch_class=EcgBatch)
dtst.split([0.9, 0.1])

#[3, 5, 8, 11, 14, 16]
#[3, 5, 11, 14, 17, 19]
all_states = {
        "QRS_model_6":     [1, 2, 3, 4, 5, 6],
        "QRS_model_9":     [3, 4, 5, 6, 7, 8],
        "QRS_model_12": [3, 4, 6, 8, 10, 11],
        "QRS_model_16":    [3, 5, 8, 11, 14, 16],
        "QRS_model_18":    [3, 5, 11, 14, 17, 19],
        "QRS_model_9_ST": [1, 4, 5, 6, 7, 8],
        "QRS_model_9_T": [1, 2, 5, 6, 7, 8],
        "QRS_model_9_ISO": [1, 2, 3, 6, 7, 8],
        "QRS_model_9_P": [1, 2, 3, 4, 7, 8],
        "QRS_model_9_PQ": [1, 2, 3, 4, 5, 8],

        "QRS_model_11_ST": [3, 6, 7, 8, 9, 10],
        "QRS_model_11_T": [3, 4, 7, 8, 9, 10],
        "QRS_model_11_ISO": [3, 4, 5, 8, 9, 10],
        "QRS_model_11_P": [3, 4, 5, 6, 9, 10],
        "QRS_model_11_PQ": [3, 4, 5, 6, 7, 10],

        "QRS_model_18_ES": [3, 6, 9, 12, 15, 18],

        # "QRS_model_22": [3, 5, 13, 18, 21, 23],
              }
type_states = {0: "QRS", 1: "ST", 2: "T", 3: "ISO", 4: "P", 5: "PQ"}
process_states_pipeline = LoadEcgPipeline(batch_size=20, annot_ext="pu1")
for model_name in all_states.keys():
    process_states_pipeline_for_model = process_states_pipeline + HMM_predict_pipeline("models\\3.2\\" + model_name + ".dill", annot="hmm_annotation" + model_name)
    batch = (dtst.test >> process_states_pipeline_for_model).run(batch_size=20, shuffle=False, drop_last=False, n_epochs=1, lazy=True).next_batch()
    print_metrics(batch, model_name, all_states[model_name], type_states)
    #for type_state in type_states:
    #    print(type_states[type_state])
    """
        states = all_states[model_name]
        parameters = calculate_old_metrics(batch,  np.array(list(states), np.int64), type_state, "hmm_annotation" + model_name)
        print(model_name + " \tsensitivity= " + str(parameters["sensitivity"]) + "\tspecificity= " + str(parameters["specificity"]))
    #"""
    """
        states = all_states[model_name]
        parameters = calculate_metrics(batch, np.array(list(states), np.int64), type_state,
                                       "hmm_annotation" + model_name)
        print(model_name +
              " \taccuracy= " + str(parameters["accuracy"]) +
              " \tprecision= " + str(parameters["precision"]) +
              " \trecall= " + str(parameters["recall"]) +
              " \tf-score= " + str(parameters["f-score"]))
    #"""
"""
process_states_pipeline_for_model = process_states_pipeline + PanTompkinsPipeline(annot="pan_tomp_annotation")
batch = (dtst.test >> process_states_pipeline_for_model).run(batch_size=20, shuffle=False, drop_last=False, n_epochs=1, lazy=True).next_batch()
#"""
"""
parameters = calculate_old_metrics(batch,  np.array(list([1]), np.int64), 0, "pan_tomp_annotation")
print("Pan-Tompkins" + "\tsensitivity= " + str(parameters["sensitivity"]) + "\tspecificity= " + str(parameters["specificity"]))
#"""
"""
parameters = calculate_metrics(batch,  np.array(list([1]), np.int64), 0, "pan_tomp_annotation")
print("Pan-Tompkins" +
              " \taccuracy= " + str(parameters["accuracy"]) +
              " \tprecision= " + str(parameters["precision"]) +
              " \trecall= " + str(parameters["recall"]) +
              " \tf-score= " + str(parameters["f-score"]))
#"""
"""
process_states_pipeline_for_model = process_states_pipeline + HilbertTransformPipeline(annot="hilbert_annotation")
batch = (dtst.test >> process_states_pipeline_for_model).run(batch_size=20, shuffle=False, drop_last=False, n_epochs=1, lazy=True).next_batch()
#"""
"""
parameters = calculate_old_metrics(batch,  np.array(list([1]), np.int64), 0, "hilbert_annotation")
print("Hilbert-transform" + "\tsensitivity= " + str(parameters["sensitivity"]) + "\tspecificity= " + str(parameters["specificity"]))
#"""
"""
parameters = calculate_metrics(batch,  np.array(list([1]), np.int64), 0, "hilbert_annotation")
print("Hilbert-transform" +
              " \taccuracy= " + str(parameters["accuracy"]) +
              " \tprecision= " + str(parameters["precision"]) +
              " \trecall= " + str(parameters["recall"]) +
              " \tf-score= " + str(parameters["f-score"]))
#"""
#process_states_pipeline = process_states_pipeline + PanTompkinsPipeline(annot="pan_tomp_annotation") + HilbertTransformPipeline(annot="hilbert_annotation")
#batch = (dtst.test >> process_states_pipeline).run(batch_size=20, shuffle=False, drop_last=False, n_epochs=1, lazy=True).next_batch()
"""
for type_state in type_states:
    print(type_states[type_state])
    for model_name in all_states.keys():
        states = all_states[model_name]
        parameters = calculate_sensitivity(batch,  np.array(list(states), np.int64), type_state, "hmm_annotation" + model_name)
        print(model_name + " \tsensitivity= " + str(parameters["sensitivity"]) + "\tspecificity= " + str(parameters["specificity"]))
"""
"""
parameters = calculate_sensitivity(batch,  np.array(list([1]), np.int64), 0, "pan_tomp_annotation")
print("Pan-Tompkins" + "\tsensitivity= " + str(parameters["sensitivity"]) + "\tspecificity= " + str(parameters["specificity"]))
parameters = calculate_sensitivity(batch,  np.array(list([1]), np.int64), 0, "hilbert_annotation")
print("Hilbert-transform" + "\tsensitivity= " + str(parameters["sensitivity"]) + "\tspecificity= " + str(parameters["specificity"]))
"""


"""
-------------------------------------------------------------------
    #show example Annotated ECG
    batch.my_show_ecg(np.array([
        np.array(list(range(states[0])), np.int64),
        #np.array(list(range(states[0], states[1])), np.int64),
        np.array(list(range(states[1], states[2])), np.int64),
        np.array(list(range(states[3], states[4])), np.int64)
        ]),
        'sel100', start=12, end=17, annot="hmm_annotation")
"""