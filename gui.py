import warnings
from tkinter.tix import *

import h5py
import matplotlib
import self as self
from matplotlib import pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pandas import np

matplotlib.use('TkAgg')
from tkinter import *


warnings.simplefilter("ignore")


def graph(metric, normal=False):
    metric_results = []

    for name in names:
        if normal is True:
            model_metricString = h5fResults[name + metric]
        else:
            model_metricString = h5fNonNormalizedResults[name + metric]
        model_metric = np.array(model_metricString)
        metric_results.append(model_metric)

    fig = Figure(figsize=(5, 3), dpi=100)

    if normal:

        fig.suptitle('Comparison of ' + metric + " for normalized data", fontweight='bold',
                     horizontalalignment='center', verticalalignment='top', y=0.999)
    else:
        fig.suptitle('Comparison of ' + metric + " for not normalized data", fontweight='bold',
                     horizontalalignment='center', verticalalignment='top', y=0.999)

    ax = fig.add_subplot(111)

    ax.boxplot(metric_results, vert=False, whis=1.5,
               positions=None, widths=None, patch_artist=False, )
    ax.set_yticklabels(names)
    canvas = FigureCanvasTkAgg(fig, master=top)
    canvas.get_tk_widget().grid(row=0, column=1, padx=20, pady=20, sticky=E)
    canvas.draw()


if __name__ == '__main__':
    top = Tk()
    top.title('Classifier comparsion app')
    top.geometry("700x400")
    top.resizable(0, 0)
    h5Results = "output/results.h5"
    h5fResults = h5py.File(h5Results, "r")
    h5NonNormalizedResults = "output/resultsNonNormalized.h5"
    h5fNonNormalizedResults = h5py.File(h5NonNormalizedResults, "r")
    names = ["RFC", "SVC", "KNN", "LR", "LDA", "NB", "CART", "ABC", "QDA"]

    metrics = ["_acc", "_err", "_fit_time", "_score_time"]

    labelframe = LabelFrame(top)
    labelframe.grid(row=0, column=0, padx=20, pady=130)

    variable = StringVar()
    variable.set(metrics[0])  # default value

    w = OptionMenu(labelframe, variable, *metrics)
    w.grid(row=2, column=0)


    var1 = BooleanVar()
    var1.set(False)
    Checkbutton(labelframe, text="Normalize data", variable=var1).grid(row=3, column=0)

    Button(labelframe, text="graph", command=lambda: graph(variable.get(), normal=var1.get())).grid(row=4, column=0)

    mainloop()

    top.mainloop()

    h5fResults.close()
    h5fNonNormalizedResults.close()
