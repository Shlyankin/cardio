import os
import cardio.batchflow as bf
import warnings
import numpy as np
from cardio import EcgBatch
from my_tools import calculate_sensitivity
from PanTompkinsAlgorithm import testPipeline, PanTompkinsPipeline

warnings.filterwarnings('ignore')

SIGNALS_PATH = "data\\qt-database-1.0.0"  # set path to QT database
SIGNALS_MASK = os.path.join(SIGNALS_PATH, "*.hea")

index = bf.FilesIndex(path=SIGNALS_MASK, no_ext=True, sort=True)
dtst = bf.Dataset(index, batch_class=EcgBatch)
dtst.split([0.1, 0.9])

pipeline = PanTompkinsPipeline(batch_size=len(dtst.train.indices), annot="pan_tomp_annotation")
ppl_inits = (dtst.train >> pipeline).run()
batch : EcgBatch = ppl_inits.next_batch(len(dtst.train.indices))
parameters = calculate_sensitivity(batch,  np.array(list([1]), np.int64), 0, "pan_tomp_annotation")
print("Pan-Tompkins" + "\tsensitivity= " + str(parameters["sensitivity"]) + "\tspecificity= " + str(parameters["specificity"]))

print("end")
