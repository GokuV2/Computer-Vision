## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, count_fingers.py
## 3. In this challenge, you are only permitted to import numpy, and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
import sys
sys.path.append('../util')
from data_handling import breakIntoGrids, reshapeIntoImage

def count_fingers(im):
    
    im = im > 92
    X = breakIntoGrids(im, s = 9)
    treeRule1 = lambda X: np.logical_and(np.logical_and(X[:, 40] == 1, X[:,0] == 0), X[:, 53] == 0)
    yhat = treeRule1(X)

    yhat_reshaped = reshapeIntoImage(yhat, im.shape)
    axis_y,axis_x = yhat_reshaped.shape
    soln = 0
    labels = [1, 2, 3]
    for i in range(axis_x):
        if((yhat_reshaped[15][i])>0):
            soln=soln+1
    if (soln<5):
        num = 0
    elif ( (soln>5) and (soln<10)):
        num = 1
    else:
        num = 2
    
    return labels[num]