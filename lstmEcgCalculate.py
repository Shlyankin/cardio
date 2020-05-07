from keras.engine.saving import load_model
import wfdb
import os
import numpy as np
import glob, os
import math
import scipy.stats as st
from os.path import basename
from my_tools import calc_metr, show_ecg, get_meta
np.random.seed(0)
annot_type = "pu"

def get_ecg_record(datfile, start=0, end=None):
    recordname = os.path.basename(datfile).split(".dat")[0]
    recordpath = os.path.dirname(datfile)
    cwd = os.getcwd()
    os.chdir(recordpath)  ## somehow it only works if you chdir.
    record = wfdb.rdsamp(recordname, sampfrom=start, sampto=end)
    record = np.transpose(record[0])
    return np.transpose(record)

def my_get_ecg_data(datfile):
    ## convert .dat/q1c to numpy arrays
    recordname = os.path.basename(datfile).split(".dat")[0]
    recordpath = os.path.dirname(datfile)
    cwd = os.getcwd()
    os.chdir(recordpath)  ## somehow it only works if you chdir.

    annotator = annot_type
    annotation = wfdb.rdann(recordname, extension=annotator, sampfrom=0, sampto=None, pb_dir=None)  # read annotation
    Lstannot = list(zip(annotation.sample, annotation.symbol, annotation.aux_note))

    FirstLstannot = min(i[0] for i in Lstannot)  # annot start
    LastLstannot = max(i[0] for i in Lstannot) - 1  # annot end
    print("first-last annotation:", FirstLstannot, LastLstannot)

    record = wfdb.rdsamp(recordname, sampfrom=FirstLstannot, sampto=LastLstannot)  # read signal
    annotation = wfdb.rdann(recordname, annotator, sampfrom=FirstLstannot,
                            sampto=LastLstannot)  ## get annotation between first and last.
    # this make annotation from 0 to end
    annotation2 = wfdb.Annotation(record_name=recordname, extension=annot_type,
                                  sample=(annotation.sample - FirstLstannot),
                                  symbol=annotation.symbol, aux_note=annotation.aux_note)

    Vctrecord = np.transpose(record[0])
    VctAnnotationHot = np.zeros((6, len(Vctrecord[1])),
                                dtype=np.int)  # 6 annotation type: 0: P, 1: PQ, 2: QR, 3: RS, 4: ST, 5: ISO (TP)
    #VctAnnotationHot[3] = 1  ## inverse of the others
    # print("ecg, 2 lead of shape" , Vctrecord.shape)
    # print("VctAnnotationHot of shape" , VctAnnotationHot.shape)
    # print('plotting extracted signal with annotation')
    # wfdb.plotrec(record, annotation=annotation2, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')

    VctAnnotations = list(zip(annotation2.sample, annotation2.symbol))  ## zip coordinates + annotations (N),(t) etc)
    # print(VctAnnotations)
    annsamp = []
    anntype = []
    for i in range(len(VctAnnotations)):
        annsamp.append(VctAnnotations[i][0])
        anntype.append(VctAnnotations[i][1])
    length = len(annsamp)
    begin = -1
    end = -1
    s = 'none'
    states = {'N': 0, 'st': 1, 't': 2, 'iso': 3, 'p': 4, 'pq': 5}
    annot_expand = -1 * np.ones(length)

    for j, samp in enumerate(annsamp):
        if anntype[j] == '(':
            begin = samp
            if (end > 0) & (s != 'none'):
                if s == 'N':
                    VctAnnotationHot[1][end:begin] = 1
                    #annot_expand[end:begin] = states['st']
                elif s == 't':
                    VctAnnotationHot[3][end:begin] = 1
                    #annot_expand[end:begin] = states['iso']
                elif s == 'p':
                    VctAnnotationHot[5][end:begin] = 1
                    #annot_expand[end:begin] = states['pq']
        elif anntype[j] == ')':
            end = samp
            if (begin >= 0) & (s != 'none'):
                VctAnnotationHot[states[s]][begin:end] = 1
                #annot_expand[begin:end] = states[s]
        else:
            s = anntype[j]
    Vctrecord = np.transpose(Vctrecord)  # transpose to (timesteps,feat)
    VctAnnotationHot = np.transpose(VctAnnotationHot)
    os.chdir(cwd)
    return Vctrecord, VctAnnotationHot

def remove_seq_gaps(x, y):
    # remove parts that are not annotated <- not ideal, but quickest for now.
    window = 150
    c = 0
    cutout = []
    include = []
    print("filterering.")
    print("before shape x,y", x.shape, y.shape)
    for i in range(y.shape[0]):
        c = c + 1
        if c < window:
            include.append(i)
        if sum(y[i, 0:5]) > 0:
            c = 0
        if c >= window:
            # print ('filtering')
            pass
    x, y = x[include, :], y[include, :]
    print(" after shape x,y", x.shape, y.shape)
    return (x, y)
# разделияет последовательность x на участки длинной и с перекрытием в o с соседом слева и справа
def splitseq(x, n, o):
    # split seq; should be optimized so that remove_seq_gaps is not needed.
    upper = math.ceil(x.shape[0] / n) * n
    print("splitting on", n, "with overlap of ", o, "total datapoints:", x.shape[0], "; upper:", upper)
    for i in range(0, upper, n):
        # print(i)
        if i == 0:
            padded = np.zeros((o + n + o, x.shape[1]))  ## pad with 0's on init
            padded[o:, :x.shape[1]] = x[i:i + n + o, :]
            xpart = padded
        else:
            xpart = x[i - o:i + n + o, :]
        if xpart.shape[0] < i:
            padded = np.zeros((o + n + o, xpart.shape[1]))  ## pad with 0's on end of seq
            padded[:xpart.shape[0], :xpart.shape[1]] = xpart
            xpart = padded

        xpart = np.expand_dims(xpart, 0)  ## add one dimension; so that you get shape (samples,timesteps,features)
        try:
            xx = np.vstack((xx, xpart))
        except UnboundLocalError:  ## on init
            xx = xpart
    print("output: ", xx.shape)
    return (xx)

def normalizesignal(x):
    x = st.zscore(x, ddof=0)
    return x


def normalizesignal_array(x):
    for i in range(x.shape[0]):
        x[i] = st.zscore(x[i], axis=0, ddof=0)
    return x

def LoaddDatFiles(datfiles):
    for datfile in datfiles:
        print(datfile)
        if basename(datfile).split(".", 1)[0] in exclude: # continue exlude files
            continue
        qf = os.path.splitext(datfile)[0] + '.' + annot_type # set annot filename
        if os.path.isfile(qf):
            # print("yes",qf,datfile)
            x, y = my_get_ecg_data(datfile)
            x, y = remove_seq_gaps(x, y)

            x, y = splitseq(x, 1000, 0), splitseq(y, 1000, 0)  ## create equal sized numpy arrays of n size and overlap of o

            x = normalizesignal_array(x)
            ## todo; add noise, shuffle leads etc. ?
            try:  ## concat
                xx = np.vstack((xx, x))
                yy = np.vstack((yy, y))
            except NameError:  ## if xx does not exist yet (on init)
                xx = x
                yy = y
    return (xx, yy)

def convertToStandard(annotated):
    batch = np.zeros(annotated.shape[0] * annotated.shape[1], dtype=np.int32)
    for i in range(annotated.shape[0]):
        for j in range(annotated.shape[1]):
            batch[i*annotated.shape[1] + j] = np.argmax(annotated[i][j])
    return batch



qtdbpath = "data\\qt-database-1.0.0\\"  ## first argument = qtdb database from physionet.
percv = 0.1  # percentage validation
exclude = set()
"""exclude.update(
    ["sel35", "sel36", "sel37", "sel50", "sel102", "sel104", "sel221", "sel232", "sel310"])  # no P annotated:"""
datfiles = glob.glob(qtdbpath + "*.dat")
#xxv, yyv = LoaddDatFiles(datfiles[-round(len(datfiles) * percv):])  ## validation data.

epochs = 10
model_name = 'new_model_' + annot_type + "_" + str(epochs) + '.h5'
model = load_model(model_name)
"""yy_predicted = model.predict(xxv)
batch = convertToStandard(yy_predicted)
annot = convertToStandard(yyv)
print(calc_metr(batch, annot, type=None))
print(calc_metr(batch, annot, type='micro'))
print(calc_metr(batch, annot, type='macro'))"""
x = get_ecg_record(qtdbpath+"sel100")
xpred = splitseq(x, 1000, 0)
xpred = normalizesignal_array(xpred)
y = model.predict(xpred)
y = convertToStandard(y)
x = x.transpose()
show_ecg(x, y, ["QRS", "ST", "T", "ISO", "P", "PQ"], [1, 2, 3, 4, 5, 6], get_meta('sel100', annot_type), 10, 15)
"""show_ecg(batch.signal[0], batch.get(component="hmm_annotation" + model_name),
             type_states.items(), all_states[model_name], batch.meta, 10, 15)"""
#print(calc_metr(batch, annot, type='samples'))
#batch = convertToStandard(yy_predicted)
# convert to default annotations
# type_states = {0: "QRS", 1: "ST", 2: "T", 3: "ISO", 4: "P", 5: "PQ"}
# now 0: P, 1: PQ, 2: QR, 3: RS, 4: ST, 5: ISO (TP)
# calculate metrics
"""
lstm_10 epochs
{'accuracy': 0.8476222222222223, 
                        QRS             ST          T           ISO         P           PQ
'precision':    array([0.90837859, 0.72005018, 0.91889856, 0.87310935, 0.83423157, 0.69812122]), 
'recall':       array([0.855489  , 0.8704021 , 0.77278177, 0.89714951, 0.84311775, 0.84308419]), 
'f-score':      array([0.88114085, 0.78811945, 0.83952984, 0.8849662 , 0.83865112, 0.76378523])}
micro
{'accuracy': 0.8476222222222223, 
'precision': 0.8476222222222223, 
'recall': 0.8476222222222223, 
'f-score': 0.8476222222222224}
macro
{'accuracy': 0.8476222222222223, 
'precision': 0.8254649123445515, 
'recall': 0.8470040531892561, 
'f-score': 0.8326987808663464}

lstm pu 10
{'accuracy': 0.843912, 
                        QRS             ST          T           ISO         P           PQ
'precision':    array([0.9012835 , 0.6625749 , 0.89377781, 0.85368763, 0.89328468, 0.81757406]), 
'recall':       array([0.84777877, 0.88405916, 0.77535334, 0.89727721, 0.86811528, 0.75662682]), 
'f-score':      array([0.87371277, 0.75745831, 0.83036448, 0.87493985, 0.88052016, 0.78592061])}
{'accuracy': 0.843912, 
'precision': 0.843912, 
'recall': 0.843912, 
'f-score': 0.843912}

{'accuracy': 0.843912, 
'precision': 0.8370304299320658, 
'recall': 0.8382017636343106,
'f-score': 0.8338193614994364}
"""