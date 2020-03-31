from keras.engine.saving import load_model
import wfdb
import os
import numpy as np
import glob, os
import math
import scipy.stats as st
from os.path import basename

def get_ecg_data(datfile):
    ## convert .dat/q1c to numpy arrays
    recordname = os.path.basename(datfile).split(".dat")[0]
    recordpath = os.path.dirname(datfile)
    cwd = os.getcwd()
    os.chdir(recordpath)  ## somehow it only works if you chdir.

    annotator = annot_type
    annotation = wfdb.rdann(recordname, extension=annotator, sampfrom=0, sampto=None, pb_dir=None) #read annotation
    Lstannot = list(zip(annotation.sample, annotation.symbol, annotation.aux_note))

    FirstLstannot = min(i[0] for i in Lstannot) # annot start
    LastLstannot = max(i[0] for i in Lstannot) - 1 # annot end
    print("first-last annotation:", FirstLstannot, LastLstannot)

    record = wfdb.rdsamp(recordname, sampfrom=FirstLstannot, sampto=LastLstannot)  # read signal
    annotation = wfdb.rdann(recordname, annotator, sampfrom=FirstLstannot,
                            sampto=LastLstannot)  ## get annotation between first and last.
    # this make annotation from 0 to end
    annotation2 = wfdb.Annotation(record_name=recordname, extension=annot_type, sample=(annotation.sample - FirstLstannot),
                                  symbol=annotation.symbol, aux_note=annotation.aux_note)

    Vctrecord = np.transpose(record[0])
    VctAnnotationHot = np.zeros((6, len(Vctrecord[1])), dtype=np.int) # 6 annotation type: 0: P, 1: PQ, 2: QR, 3: RS, 4: ST, 5: ISO (TP)
    VctAnnotationHot[5] = 1  ## inverse of the others
    # print("ecg, 2 lead of shape" , Vctrecord.shape)
    # print("VctAnnotationHot of shape" , VctAnnotationHot.shape)
    # print('plotting extracted signal with annotation')
    # wfdb.plotrec(record, annotation=annotation2, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')

    VctAnnotations = list(zip(annotation2.sample, annotation2.symbol))  ## zip coordinates + annotations (N),(t) etc)
    # print(VctAnnotations)
    for i in range(len(VctAnnotations)):
        # print(VctAnnotations[i]) # Print to display annotations of an ecg
        try:

            if VctAnnotations[i][1] == "p":
                if VctAnnotations[i - 1][1] == "(":
                    pstart = VctAnnotations[i - 1][0]
                if VctAnnotations[i + 1][1] == ")":
                    pend = VctAnnotations[i + 1][0]
                if VctAnnotations[i + 3][1] == "N":
                    rpos = VctAnnotations[i + 3][0]
                    if VctAnnotations[i + 2][1] == "(":
                        qpos = VctAnnotations[i + 2][0]
                    if VctAnnotations[i + 4][1] == ")":
                        spos = VctAnnotations[i + 4][0]
                    for ii in range(0, 8):  ## search for t (sometimes the "(" for the t  is missing  )
                        if VctAnnotations[i + ii][1] == "t":
                            if VctAnnotations[i + ii - 1][1] == "(":
                                tstart = VctAnnotations[i + ii - 1][0]
                            else:
                                tstart = VctAnnotations[i + ii][0]
                            if VctAnnotations[i + ii + 1] == ")":
                                tend = VctAnnotations[i + ii + 1][0]

                                VctAnnotationHot[0][qpos:spos] = 1 #QRS
                                VctAnnotationHot[1][spos:tstart] = 1 #ST
                                VctAnnotationHot[2][tstart:tend] = 1 #T
                                VctAnnotationHot[3][tend:pstart] = 1 #ISO
                                VctAnnotationHot[4][pstart:pend] = 1 #P
                                VctAnnotationHot[5][pend:qpos] = 1 # PQ

        except IndexError:
            pass

    Vctrecord = np.transpose(Vctrecord)  # transpose to (timesteps,feat)
    VctAnnotationHot = np.transpose(VctAnnotationHot)
    os.chdir(cwd)
    return Vctrecord, VctAnnotationHot