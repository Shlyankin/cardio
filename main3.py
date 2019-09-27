import os
import cardio.batchflow as bf
import warnings
import numpy as np
from cardio import EcgBatch
from PanTompkinsAlgorithm import testPipeline, PanTompkinsPipeline

warnings.filterwarnings('ignore')

SIGNALS_PATH = "data\\qt-database-1.0.0"  # set path to QT database
SIGNALS_MASK = os.path.join(SIGNALS_PATH, "*.hea")

index = bf.FilesIndex(path=SIGNALS_MASK, no_ext=True, sort=True)
dtst = bf.Dataset(index, batch_class=EcgBatch)
dtst.split([0.1, 0.9])

pipeline = PanTompkinsPipeline(batch_size=10)
ppl_inits = (dtst.train >> pipeline).run()
t = ppl_inits.get_variable("qrs_annotation")
print("end")
