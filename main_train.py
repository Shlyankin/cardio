import os
import cardio.batchflow as bf
from cardio import EcgBatch
from my_pipelines.my_pipelines import LoadEcgPipeline, HMM_preprocessing_pipeline, HMM_train_pipeline

# Create dataset
SIGNALS_PATH = "data\\qt-database-1.0.0"  # set path to QT database
SIGNALS_MASK = os.path.join(SIGNALS_PATH, "*.hea")
index = bf.FilesIndex(path=SIGNALS_MASK, no_ext=True, sort=True)
dtst = bf.Dataset(index, batch_class=EcgBatch)
dtst.split([0.9, 0.1])

# Training models and they states
all_states = {"QRS_model_6":  [1, 2,  3,  4,  5,  6],
              "QRS_model_9":  [3, 4,  5,  6,  7,  8],
              "QRS_model_16": [3, 5,  8, 11, 14, 16],
              "QRS_model_18": [3, 5, 11, 14, 17, 19],
              "QRS_model_22": [3, 5, 13, 18, 21, 23]}

# Preprocess data
pipeline = LoadEcgPipeline() + HMM_preprocessing_pipeline()
ppl_inits = (dtst.train >> pipeline).run(batch_size=95, shuffle=False, drop_last=False, n_epochs=1)

# Train and save models
for model_name in all_states.keys():
    pipeline = HMM_train_pipeline(ppl_inits, states=all_states[model_name])
    ppl_train = (dtst.train >> pipeline).run(batch_size=95, shuffle=False, drop_last=False, n_epochs=1)
    ppl_train.save_model("HMM", path=model_name + ".dill")
    print(model_name + " is trained")
