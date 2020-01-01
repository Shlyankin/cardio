import os
import cardio.batchflow as bf
import time
from cardio import EcgBatch
from my_pipelines.my_pipelines import LoadEcgPipeline, HMM_preprocessing_pipeline, HMM_train_pipeline

# Create dataset
SIGNALS_PATH = "data\\qt-database-1.0.0"  # set path to QT database
SIGNALS_MASK = os.path.join(SIGNALS_PATH, "*.hea")
index = bf.FilesIndex(path=SIGNALS_MASK, no_ext=True, sort=True)
dtst = bf.Dataset(index, batch_class=EcgBatch)
dtst.split([0.9, 0.1])

# Training models and they states
# {0: "QRS", 1: "ST", 2: "T", 3: "ISO", 4: "P", 5: "PQ"}
all_states = {
    #"QRS_model_6":      [1, 2, 3, 4, 5, 6],
    #"QRS_model_8":      [3, 4, 5, 6, 7, 8],
    #"QRS_model_11":     [3, 4, 6, 8, 10, 11],
    #"QRS_model_14":     [3, 5, 8, 10, 13, 14],
    "QRS_model_16":     [3, 5, 8, 11, 14, 16],
    "QRS_model_19":     [3, 5, 11, 14, 17, 19]
    #"QRS_model_8_ST":   [1, 4, 5, 6, 7, 8],
    #"QRS_model_8_T":    [1, 2, 5, 6, 7, 8],
    #"QRS_model_8_ISO":  [1, 2, 3, 6, 7, 8],
    #"QRS_model_8_P":    [1, 2, 3, 4, 7, 8],
    #"QRS_model_8_PQ":   [1, 2, 3, 4, 5, 8],

    #"QRS_model_10_ST":  [3, 6, 7, 8, 9, 10],
    #"QRS_model_10_T":   [3, 4, 7, 8, 9, 10],
    #"QRS_model_10_ISO": [3, 4, 5, 8, 9, 10],
    #"QRS_model_10_P":   [3, 4, 5, 6, 9, 10],
    #"QRS_model_10_PQ":  [3, 4, 5, 6, 7, 10],

    #"QRS_model_18_ES":  [3, 6, 9, 12, 15, 18],

    #"QRS_model_23":     [3, 5, 13, 18, 21, 23],
}

# Preprocess data
pipeline = LoadEcgPipeline() + HMM_preprocessing_pipeline()
ppl_inits = (dtst.train >> pipeline).run(batch_size=95, shuffle=False, drop_last=False, n_epochs=1)

# Train and save models
for model_name in all_states.keys():
    st_time = time.time()
    pipeline = HMM_train_pipeline(ppl_inits, states=all_states[model_name], n_iter=40)
    ppl_train = (dtst.train >> pipeline).run(batch_size=95, shuffle=False, drop_last=False, n_epochs=1)
    ppl_train.save_model("HMM", path=model_name + ".dill")
    end_time = time.time()
    f = open("result_time.txt", "a")
    f.write(model_name + " is trained for " + str((end_time - st_time)/60) + " min\n")
    f.close()
    print(model_name + " is trained for " + str((end_time - st_time)/60) + " min")
os.system('shutdown -s')
