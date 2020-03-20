# AI

We're using 9 models:  
1. Random Forest Classifier ```RFC```  
2. SVC ```SVC```  
3. KN eighbors Classifier ```KNN```  
4. Logistic Regression ```LR```  
5. Linear Discriminant Analysis ```LDA```  
6. Gaussian Naive Bayes ```NB```  
7. Decision Tree Classifier ```CART```  
8. Ada Boost Classifier ```ABC```  
9. Quadratic Discriminant Analysis ```QDA```  

For each model we save cross validation score based on <b>accuracy</b> and <b>mean squared error</b> as well as <b>fit time</b> and <b>score time</b>.  
* Accuracy: ```_acc```
* Mean squared error: ```_err```
* Fit time: ```_fit_time```
* Score time: ```_score_time```  

All of this is saved into <i>resultsNonNormalized.h5</i> or <i>results.h5</i>, where the names indicate whether we fed the ml model normalized or non normalized data.  
The access the data we need to load the .h5 file and then read correct dataset from it, so we do:  
```python
import h5py
import numpy as np

h5Results = "output/results.h5"
h5fResults = h5py.File(h5Results, "r")
data = np.array(h5fResults["RFC_fit_time"])  # fit time for random forest classifier
```  
To read the correct dataset we take given model's short name (RFC, SVC etc,) and add the postfix of what we want (_acc, _err, etc.)  
