import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter

def plot_metrics():
    data = {
        "models": {
        "QRS_model_6":  {"accuracy": [0.8754], "precision": [0.6254], "recall": [0.6267], "f-score": [0.6261]},
        "QRS_model_9":  {"accuracy": [0.8782], "precision": [0.6337], "recall": [0.6350], "f-score": [0.6343]},
        "QRS_model_12": {"accuracy": [0.8743], "precision": [0.6221], "recall": [0.6234], "f-score": [0.6227]},
        "QRS_model_16": {"accuracy": [0.9025], "precision": [0.7065], "recall": [0.7079], "f-score": [0.7072]},
        "QRS_model_18": {"accuracy": [0.9016], "precision": [0.7039], "recall": [0.7053], "f-score": [0.7046]},
        "QRS_model_22": {"accuracy": [0.9031], "precision": [0.7084], "recall": [0.7098], "f-score": [0.7091]}
        },
        "QRS": {
            "QRS_model_6":  {"accuracy": [0.9197], "precision": [0.6356], "recall": [0.9165], "f-score": [0.7506]},
            "QRS_model_9":  {"accuracy": [0.9035], "precision": [0.5822], "recall": [0.9505], "f-score": [0.7221]},
            "QRS_model_12": {"accuracy": [0.9345], "precision": [0.7013], "recall": [0.8765], "f-score": [0.7792]},
            "QRS_model_16": {"accuracy": [0.9425], "precision": [0.7496], "recall": [0.8464], "f-score": [0.7951]},
            "QRS_model_18": {"accuracy": [0.9543], "precision": [0.8161], "recall": [0.8431], "f-score": [0.8294]},
            "QRS_model_22": {"accuracy": [0.9547], "precision": [0.8220], "recall": [0.8380], "f-score": [0.8300]}
        },
        "ISO": {
            "QRS_model_6":  {"accuracy": [0.8119], "precision": [0.8329], "recall": [0.6307], "f-score": [0.7178]},
            "QRS_model_9":  {"accuracy": [0.7996], "precision": [0.8469], "recall": [0.5759], "f-score": [0.6856]},
            "QRS_model_12": {"accuracy": [0.7739], "precision": [0.7695], "recall": [0.5768], "f-score": [0.6594]},
            "QRS_model_16": {"accuracy": [0.8317], "precision": [0.8194], "recall": [0.7137], "f-score": [0.7629]},
            "QRS_model_18": {"accuracy": [0.8369], "precision": [0.8862], "recall": [0.6540], "f-score": [0.7526]},
            "QRS_model_22": {"accuracy": [0.8466], "precision": [0.8706], "recall": [0.6998], "f-score": [0.7759]}
        },
        "P": {
            "QRS_model_6":  {"accuracy": [0.8907], "precision": [0.3049], "recall": [0.6659], "f-score": [0.4183]},
            "QRS_model_9":  {"accuracy": [0.9058], "precision": [0.3618], "recall": [0.7804], "f-score": [0.4944]},
            "QRS_model_12": {"accuracy": [0.8581], "precision": [0.2646], "recall": [0.7897], "f-score": [0.3964]},
            "QRS_model_16": {"accuracy": [0.9186], "precision": [0.4010], "recall": [0.7673], "f-score": [0.5267]},
            "QRS_model_18": {"accuracy": [0.9084], "precision": [0.3696], "recall": [0.7823], "f-score": [0.5020]},
            "QRS_model_22": {"accuracy": [0.9111], "precision": [0.3643], "recall": [0.6794], "f-score": [0.4743]}
        },
        "PQ": {
            "QRS_model_6":  {"accuracy": [0.9380], "precision": [0.3120], "recall": [0.4930], "f-score": [0.3821]},
            "QRS_model_9":  {"accuracy": [0.9452], "precision": [0.3856], "recall": [0.6875], "f-score": [0.4941]},
            "QRS_model_12": {"accuracy": [0.9537], "precision": [0.4124], "recall": [0.4468], "f-score": [0.4289]},
            "QRS_model_16": {"accuracy": [0.9465], "precision": [0.3583], "recall": [0.4732], "f-score": [0.4078]},
            "QRS_model_18": {"accuracy": [0.9459], "precision": [0.3572], "recall": [0.4862], "f-score": [0.4118]},
            "QRS_model_22": {"accuracy": [0.9376], "precision": [0.3154], "recall": [0.5148], "f-score": [0.3912]}
        },
        "ST": {
            "QRS_model_6":  {"accuracy": [0.8553], "precision": [0.4448], "recall": [0.4424], "f-score": [0.4436]},
            "QRS_model_9":  {"accuracy": [0.8676], "precision": [0.4910], "recall": [0.4148], "f-score": [0.4497]},
            "QRS_model_12": {"accuracy": [0.8915], "precision": [0.7480], "recall": [0.2539], "f-score": [0.3792]},
            "QRS_model_16": {"accuracy": [0.9127], "precision": [0.6914], "recall": [0.5970], "f-score": [0.6408]},
            "QRS_model_18": {"accuracy": [0.9051], "precision": [0.7673], "recall": [0.3909], "f-score": [0.5179]},
            "QRS_model_22": {"accuracy": [0.9051], "precision": [0.7745], "recall": [0.3840], "f-score": [0.5135]}
        },
        "T": {
            "QRS_model_6":  {"accuracy": [0.8372], "precision": [0.7363], "recall": [0.5772], "f-score": [0.6471]},
            "QRS_model_9":  {"accuracy": [0.8475], "precision": [0.7410], "recall": [0.6307], "f-score": [0.6814]},
            "QRS_model_12": {"accuracy": [0.8344], "precision": [0.6612], "recall": [0.7376], "f-score": [0.6973]},
            "QRS_model_16": {"accuracy": [0.8629], "precision": [0.7490], "recall": [0.7067], "f-score": [0.7272]},
            "QRS_model_18": {"accuracy": [0.8590], "precision": [0.6732], "recall": [0.8841], "f-score": [0.7644]},
            "QRS_model_22": {"accuracy": [0.8634], "precision": [0.6891], "recall": [0.8595], "f-score": [0.7649]}
        },

    }
    models = {
        "QRS_model_6":  {"accuracy": [0.8754],	"precision": [0.6254],	"recall": [0.6267],	"f-score": [0.6261]},
        "QRS_model_9":  {"accuracy": [0.8782],	"precision": [0.6337],	"recall": [0.6350],	"f-score": [0.6343]},
        "QRS_model_12": {"accuracy": [0.8743],	"precision": [0.6221],	"recall": [0.6234],	"f-score": [0.6227]},
        "QRS_model_16": {"accuracy": [0.9025],	"precision": [0.7065],	"recall": [0.7079],	"f-score": [0.7072]},
        "QRS_model_18": {"accuracy": [0.9016],	"precision": [0.7039],	"recall": [0.7053],	"f-score": [0.7046]},
        "QRS_model_22": {"accuracy": [0.9031],	"precision": [0.7084],	"recall": [0.7098],	"f-score": [0.7091]}
    }
    for state in data:
        metrics = ["accuracy", "precision", "recall", "f-score"]
        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        for i in range(2):
            for j in range(2):
                ax: plt.Axes = axes[i][j]
                ax.set_ylim(ymin=0.0, ymax=1.0)
                for model in models.keys():
                    model_formated = model.replace('_', ' ')
                    ax.bar(model_formated, data[state][model][metrics[i*2 + j]], width=0.8, label=model_formated)
                ax.set_xlabel("models")
                ax.set_ylabel(metrics[i*2 + j])
                ax.set_title(state + " " + metrics[i*2 + j])
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
        plt.show()

def plot_time():
    times = [29.8,	45.6,	66.6,	152.1,	191.3,	491.8]
    nodes = [6,	8, 11, 16, 18, 22]

    ax1: plt.Axes
    _, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    ax1.plot(nodes, times)
    #ax1.xaxis.set_major_locator(ticker.MultipleLocator(4))
    #ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    #ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    #  Добавляем линии основной сетки:
    ax1.grid(which='major', color='k')
    ax1.minorticks_on()
    ax1.grid(which='minor', color='gray', linestyle=':')
    ax1.set_ylim(ymin=0, ymax=times[-1])
    ax1.set_xlim(xmin=nodes[0], xmax=nodes[-1])
    ax1.set_title("График зависимости времени обучения от количества состояний СММ")
    ax1.set_xlabel("Количество состояний СММ")
    ax1.set_ylabel("Время обучения, мин")
    plt.rc('legend', fontsize=22)  # legend fontsize
    plt.rc('figure', titlesize=48)  # fontsize of the figure title
    plt.show()

def plot_pp_lab1():
    times = {
        "MPI":    [1.847744e-06, 7.708867e-06, 1.568397e-04],
        "OpenMP": [5.316734e-05, 1.015862e-04, 2.466142e-03]
    }
    nodes = [1, 3, 9]

    ax1: plt.Axes
    _, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    for item in times.keys():
        ax1.plot(nodes, times[item], label=item)
    ax1.legend(shadow=True, ncol=2)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1e-03))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1e-04))
    #  Добавляем линии основной сетки:
    ax1.grid(which='major', color='k')
    ax1.minorticks_on()
    ax1.grid(which='minor', color='gray', linestyle=':')
    ax1.set_ylim(ymin=0, ymax=3e-03)
    ax1.set_xlim(xmin=nodes[0], xmax=nodes[-1])
    ax1.set_title("Параллельные алгоритмы поиска суммы элементов массива")
    ax1.set_xlabel("Количество нитей, шт")
    ax1.set_ylabel("Среднее время работы, с")
    plt.rc('legend', fontsize=22)  # legend fontsize
    plt.rc('figure', titlesize=48)  # fontsize of the figure title
    plt.show()

def plot_pp_lab2():
    times = {
        "MPI point-to-point":   [0.000494, 0.000949],
        "MPI collective op":    [0.000505, 0.000959],
        "OpenMP critical":      [1.624782, 1.256059],
        "OpenMP atomic":        [0.682222, 0.398773],
        "OpenMP reduce":        [0.004565, 0.008743]
    }
    nodes = [2, 4]

    ax1: plt.Axes
    _, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    for item in times.keys():
        ax1.plot(nodes, times[item], label=item)
    ax1.legend(shadow=True, ncol=2)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    #  Добавляем линии основной сетки:
    ax1.grid(which='major', color='k')
    ax1.minorticks_on()
    ax1.grid(which='minor', color='gray',linestyle=':')
    ax1.set_ylim(ymin=0, ymax=1.8)
    ax1.set_xlim(xmin=nodes[0], xmax=nodes[-1])
    ax1.set_title("Параллельные алгоритмы поиска суммы элементов массива")
    ax1.set_xlabel("Количество нитей, шт")
    ax1.set_ylabel("Среднее время работы, с")
    plt.rc('legend', fontsize=22)    # legend fontsize
    plt.rc('figure', titlesize=48)  # fontsize of the figure title
    plt.show()


def plot_pp_lab3():
    """
times = {
        "MPI кратно":          [0.000884,   0.008392,   0.030042],
        "MPI не кратно":       [0.000879,   0.008379,   0.030013],
        "OpenMP Static":       [0.000364,   0.000589,   0.001192],
        "OpenMP Guided":       [0.000103,   0.000403,   0.000783],
        "OpenMP Dynamic":      [0.000479,   0.001624,	0.003175],
    }

times = {
    "MPI кратно":          [0.001012, 0.009576, 0.034068],
    "MPI не кратно":       [0.000980, 0.009416, 0.033776],
    "OpenMP Static":       [0.000406, 0.000327, 0.000629],
    "OpenMP Guided":       [0.000083, 0.000309, 0.000625],
    "OpenMP Dynamic":      [0.000403, 0.001083, 0.002171],
}
0.000936, 0.008803, 0.035320
0.007908, 0.035017, 0.056273
0.001571, 0.000313, 0.000613
0.000833, 0.000949, 0.000621
0.001290, 0.000310, 0.001898
"""
    times = {
        "MPI кратно":          [0.000936, 0.008803, 0.035320],
        "MPI не кратно":       [0.007908, 0.035017, 0.056273],
        "OpenMP Static":       [0.001571, 0.000313, 0.000613],
        "OpenMP Guided":       [0.000833, 0.000949, 0.000621],
        "OpenMP Dynamic":      [0.001290, 0.000310, 0.001898],
    }
    nodes = [1, 4, 8]

    ax1: plt.Axes
    _, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    for item in times.keys():
        ax1.plot(nodes, times[item], label=item)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.001))
    #  Добавляем линии основной сетки:
    ax1.grid(which='major', color='k')
    ax1.minorticks_on()
    ax1.grid(which='minor', color='gray',linestyle=':')
    ax1.set_ylim(ymin=0, ymax=0.04)
    ax1.set_xlim(xmin=nodes[0], xmax=nodes[-1])
    ax1.set_title("Параллельные алгоритмы поиска суммы векторов")
    ax1.set_xlabel("Параметр Q")
    ax1.set_ylabel("Среднее время работы, с")
    plt.rc('legend', fontsize=22)    # legend fontsize
    plt.rc('figure', titlesize=48)  # fontsize of the figure title
    plt.show()


def plot_pp_lab4():
    """
0.000909536
0.000863872
0.000859136
0.000866720
0.000920096

"""
    times = {
        "Block = 1024":         [0.000909536],
        "Block = 512":          [0.000863872],
        "Block = 256":          [0.000859136],
        "Block = 128":          [0.000866720],
        "Block = 64":           [0.000920096],
        "MPI 16 потоков":       [0.010472],
        "OpenMP 16 потоков":    [0.005470]
    }
    block_size = [1024, 512, 256, 128, 64]

    ax1: plt.Axes
    _, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(16, 18))
    for item in times.keys():
        ax1.bar(item, times[item], width=0.8, label=item)
        #ax1.plot(block_size, times[item], label=item)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.001))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.0001))
    #  Добавляем линии основной сетки:
    ax1.yaxis.set_visible(True)
    #ax1.xaxis.set_visible(False)
    ax1.grid(which='major', color='k')
    ax1.minorticks_on()
    ax1.grid(which='minor', color='gray',linestyle=':')
    #ax1.set_ylim(ymin=0, ymax=0.001)

    ax1.set_title("Параллельные алгоритмы поиска суммы векторов")
    #ax1.set_xlabel("Размер блока, шт")
    ax1.set_ylabel("Среднее время работы, с")
    plt.rc('legend', fontsize=22)    # legend fontsize
    plt.rc('figure', titlesize=48)  # fontsize of the figure title
    plt.show()


def plot_pp_lab5():
    """
0.000909536
0.000863872
0.000859136
0.000866720
0.000920096

"""
    times = [ 18.6377, 11.7337, 6.82614, 5.10599, 2.44553,
                0.393685, 0.128063, 0.0945155, 0.091427, 0.0]
    matrix_size = [14000, 12000, 10000, 9000, 7000, 3500, 1750, 800, 400, 0]
    times.reverse()
    matrix_size.reverse()

    ax1: plt.Axes
    _, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax1.plot(matrix_size, times, label="умножение квадратных матриц типа double cublas")
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(500))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    #  Добавляем линии основной сетки:
    ax1.grid(which='major', color='k')
    ax1.minorticks_on()
    ax1.grid(which='minor', color='gray', linestyle=':')
    ax1.set_ylim(ymin=0.0, ymax=times[-1])
    ax1.set_xlim(xmin=matrix_size[0], xmax=matrix_size[-1])
    ax1.set_title("Умножение квадратных матриц типа double с помощью cublas")
    ax1.set_xlabel("Размерность матриц")
    ax1.set_ylabel("Среднее время работы, с")
    plt.rc('legend', fontsize=22)  # legend fontsize
    plt.rc('figure', titlesize=48)  # fontsize of the figure title
    plt.show()
#plot_pp_lab5()
#plot_pp_lab4()
plot_pp_lab3()
#plot_pp_lab2()
#plot_pp_lab1()
#plot_time()
#plot_metrics()
