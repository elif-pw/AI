import h5py
import numpy as np
from matplotlib import pyplot
import warnings

warnings.simplefilter("ignore")

# give one metric and specify the dataset (default is not normalized)
def graph(metric, normal=False):
    metric_results = []
    for name in names:
        if normal is True:
            model_metricString = h5fResults[name + metric]
        else:
            model_metricString = h5fNonNormalizedResults[name + metric]
        model_metric = np.array(model_metricString)
        metric_results.append(model_metric)

    fig = pyplot.figure()
    if normal:
        fig.suptitle('Comparison of ' + metric + " for normalized data")
    else:
        fig.suptitle('Comparison of ' + metric + " for not normalized data")

    ax = fig.add_subplot(111)
    pyplot.boxplot(metric_results)
    ax.set_xticklabels(names)
    pyplot.show()


if __name__ == '__main__':
    h5Results = "output/results.h5"
    h5fResults = h5py.File(h5Results, "r")
    h5NonNormalizedResults = "output/resultsNonNormalized.h5"
    h5fNonNormalizedResults = h5py.File(h5NonNormalizedResults, "r")
    names = ["RFC", "SVC", "KNN", "LR", "LDA", "NB", "CART", "ABC", "QDA"]

    metrics = ["_acc", "_err", "_fit_time", "_score_time"]
    # _score_time metric for not normalized dataset
    graph(metrics[3])

    # _fit_time metric for normalized dataset
    graph(metrics[2], normal=True)

    h5fResults.close()
    h5fNonNormalizedResults.close()
