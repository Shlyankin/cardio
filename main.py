import os
import sys
import numpy as np
import tensorflow as tf
import cardio.batchflow as bf
from cardio import EcgBatch
from cardio.models.metrics import classification_report

sys.path.append("..")
index = bf.FilesIndex(path="data/A*.hea", no_ext=True, sort=True)
print(index.indices)
eds = bf.Dataset(index, batch_class=EcgBatch)
batch = eds.next_batch(batch_size=2)
batch_with_data = batch.load(fmt="wfdb", components=["signal", "meta"])
#batch_with_data.show_ecg('A00001', start=10, end=15)

from cardio.pipelines import hmm_preprocessing_pipeline, hmm_train_pipeline
import warnings
warnings.filterwarnings('ignore')

SIGNALS_PATH = "qt-database-1.0.0" #set path to QT database
SIGNALS_MASK = os.path.join(SIGNALS_PATH, "*.hea")

index = bf.FilesIndex(path=SIGNALS_MASK, no_ext=True, sort=True)
dtst = bf.Dataset(index, batch_class=EcgBatch)

from my_pipelines.my_pipelines import HMM_train_pipeline, HMM_preprocessing_pipeline, my_HMM_train_pipeline, my_HMM_predict_pipeline
"""
pipeline = HMM_preprocessing_pipeline()
ppl_inits = (dtst >> pipeline).run()

pipeline = my_HMM_train_pipeline(ppl_inits)
ppl_train = (dtst >> pipeline).run()
ppl_train.save_model("HMM", path="QRS_model_16.dill")
"""


"""
pipeline = hmm_preprocessing_pipeline()
ppl_inits = (dtst >> pipeline).run()

pipeline = hmm_train_pipeline(ppl_inits)
ppl_train = (dtst >> pipeline).run()
"""



from my_tools import expand_annotation
from cardio.pipelines import hmm_predict_pipeline
from my_tools import calculate_sensitivity
model_name = "QRS_model_6.dill"
#[3, 5, 8, 11, 14, 16]
#[3, 5, 11, 14, 17, 19]
all_states = {"QRS_model_6.dill": [1, 2, 3, 4, 5, 6],
              "QRS_model_16.dill": [3, 5, 8, 11, 14, 16],
              "QRS_model_18.dill": [3, 5, 11, 14, 17, 19]}
for model_name in all_states.keys():
    states = all_states[model_name]
    batch:EcgBatch = (dtst >> my_HMM_predict_pipeline(model_name, annot="hmm_annotation")).next_batch()
    parameters = calculate_sensitivity(batch,  np.array(list(states), np.int64), 0, "hmm_annotation")
    print(model_name + "\tsensitivity= " + str(parameters["sensitivity"]) + "\tspecificity= " + str(parameters["specificity"]))
    batch.my_show_ecg(np.array([
        np.array(list(range(states[0])), np.int64),
        #np.array(list(range(states[0], states[1])), np.int64),
        np.array(list(range(states[1], states[2])), np.int64),
        np.array(list(range(states[3], states[4])), np.int64)
        ]),
        'sel100', start=12, end=17, annot="hmm_annotation")


