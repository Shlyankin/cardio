import os
import cardio.batchflow as bf
import warnings
import numpy as np
from cardio import EcgBatch
from my_tools import calculate_sensitivity
from PanTompkinsAlgorithm import HilbertTransformPipeline

warnings.filterwarnings('ignore')

SIGNALS_PATH = "data\\qt-database-1.0.0"  # set path to QT database
SIGNALS_MASK = os.path.join(SIGNALS_PATH, "*.hea")

index = bf.FilesIndex(path=SIGNALS_MASK, no_ext=True, sort=True)
dtst = bf.Dataset(index, batch_class=EcgBatch)
dtst.split([0.1, 0.9])

pipeline = HilbertTransformPipeline(batch_size=1, annot="hilb_annotation")
ppl_inits = (dtst.train >> pipeline).run()
batch : EcgBatch = ppl_inits.next_batch(len(dtst.train.indices))
print("end")
