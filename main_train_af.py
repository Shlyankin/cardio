import tensorflow as tf
import os
import sys
from cardio import batchflow as bf
from cardio import EcgBatch
from my_pipelines.my_pipelines import dirichlet_train_pipeline
sys.path.append("..")

index = bf.FilesIndex(path="data\\A*.hea", no_ext=True, sort=True)
eds = bf.Dataset(index, batch_class=EcgBatch)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33, allow_growth=True)

AF_SIGNALS_PATH = "data\\training2017" #set path to PhysioNet database
AF_SIGNALS_MASK = os.path.join(AF_SIGNALS_PATH, "*.hea")
AF_SIGNALS_REF = os.path.join(AF_SIGNALS_PATH, "REFERENCE.csv")

index = bf.FilesIndex(path=AF_SIGNALS_MASK, no_ext=True, sort=True)
afds = bf.Dataset(index, batch_class=EcgBatch)

pipeline = dirichlet_train_pipeline(AF_SIGNALS_REF, gpu_options=gpu_options)
train_ppl = (afds >> pipeline).run()
model_path = "af_model_dump"
train_ppl.save_model("dirichlet", path=model_path)

from cardio.pipelines import dirichlet_predict_pipeline

pipeline = dirichlet_predict_pipeline(model_path, gpu_options=gpu_options)
res = (eds >> pipeline).run()
pred = res.get_variable("predictions_list")

print(["{:.2f}".format(x["target_pred"]["A"]) for x in pred])