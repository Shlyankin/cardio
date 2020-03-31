from keras.engine.saving import load_model
import wfdb
import os
import numpy as np
import glob, os
import math
import scipy.stats as st
from os.path import basename

np.random.seed(0)
annot_type = "pu1"

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
    VctAnnotationHot[3] = 1  ## inverse of the others
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
                            if VctAnnotations[i + ii + 1][1] == ")":
                                tend = VctAnnotations[i + ii + 1][0]

                                VctAnnotationHot[0][qpos:spos] = 1  # QRS
                                VctAnnotationHot[1][spos:tstart] = 1  # ST
                                VctAnnotationHot[2][tstart:tend] = 1  # T
                                VctAnnotationHot[3][tend:pstart] = 0  # ISO
                                VctAnnotationHot[4][pstart:pend] = 1  # P
                                VctAnnotationHot[5][pend:qpos] = 1  # PQ
        except IndexError:
            pass

    Vctrecord = np.transpose(Vctrecord)  # transpose to (timesteps,feat)
    VctAnnotationHot = np.transpose(VctAnnotationHot)
    os.chdir(cwd)
    return Vctrecord, VctAnnotationHot

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
                            tpos = VctAnnotations[i + ii][0]
                            if VctAnnotations[i + ii + 1][1] == ")":
                                tendpos = VctAnnotations[i + ii + 1][0]
                                # 				#print(ppos,qpos,rpos,spos,tendpos)
                                VctAnnotationHot[0][pstart:pend] = 1  # P segment
                                VctAnnotationHot[1][
                                pend:qpos] = 1  # part "nothing" between P and Q, previously left unnanotated, but categorical probably can't deal with that
                                VctAnnotationHot[2][qpos:rpos] = 1  # QR
                                VctAnnotationHot[3][rpos:spos] = 1  # RS
                                VctAnnotationHot[4][spos:tendpos] = 1  # ST (from end of S to end of T)
                                VctAnnotationHot[5][
                                pstart:tendpos] = 0  # tendpos:pstart becomes 1, because it is inverted above
        except IndexError:
            pass

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

            x, y = splitseq(x, 1000, 150), splitseq(y, 1000,
                                                    150)  ## create equal sized numpy arrays of n size and overlap of o

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
percv = 0.01  # percentage validation
exclude = set()
exclude.update(
    ["sel35", "sel36", "sel37", "sel50", "sel102", "sel104", "sel221", "sel232", "sel310"])  # no P annotated:
datfiles = glob.glob(qtdbpath + "*.dat")
xxv, yyv = LoaddDatFiles(datfiles[-round(len(datfiles) * percv):])  ## validation data.

epochs = 10
model_name = 'new_model_' + str(epochs) + '.h5'
model = load_model(model_name)
yy_predicted = model.predict(xxv)
batch = convertToStandard(yy_predicted)
print("")
#batch = convertToStandard(yy_predicted)
# convert to default annotations
# type_states = {0: "QRS", 1: "ST", 2: "T", 3: "ISO", 4: "P", 5: "PQ"}
# now 0: P, 1: PQ, 2: QR, 3: RS, 4: ST, 5: ISO (TP)
# calculate metrics
