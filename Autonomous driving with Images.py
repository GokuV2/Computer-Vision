## ---------------------------- ##
##
## Example student submission code for autonomous driving challenge.
## You must modify the train and predict methods and the NeuralNetwork class. 
## 
## ---------------------------- ##

import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

def im_preprocess(img):
    resize_img=cv2.resize(img, (int(94), int(90)), interpolation = cv2.INTER_AREA)
    get_alv=alv_vision(resize_img, rgb = [-1, 0,1], thresh = 0.85)
    alv_1 = get_alv.ravel()
    norm_img=alv_1/255
    return norm_img

def bin_evaluation(val):
    gaus_val = np.asarray([0.1,0.3,0.5,0.7,1.0,0.7,0.5,0.3,0.1])
    
    if((val) > 55):
        
        gaus_bin = np.zeros(90)
        gaus_bin[val-4:val+5]= gaus_val[0:(gaus_val.shape[0]-((val+5)-90))]
            
    elif((val) < 4):
        gaus_bin = np.zeros(90)
        gaus_bin[0:len(gaus_val[4-val:9])]=gaus_val[4-val:9]
    
    else:
        gaus_bin = np.zeros(90)
        gaus_bin[val-4:val+5]=gaus_val
    
    return gaus_bin

def train(path_to_images, csv_file):
    
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]
    
    X_ar=[]
    for i in range(0,frame_nums.shape[0]):
        im_full = cv2.imread(path_to_images + '/' + str(int(i)).zfill(4) + '.jpg')
        processed_img = im_preprocess(im_full)
        X_ar.append(processed_img)
        
    bin_space=np.linspace(np.min(steering_angles,axis=0),np.max(steering_angles,axis=0),90)
    
    y_ar = []
    for i in range(0,1500):
        val = np.argmin(abs(np.subtract(bin_space,steering_angles[i])))
        gaus_bin = bin_evaluation(val)    
        y_ar.append(gaus_bin)
    
    X=np.asarray(X_ar)
    y=np.asarray(y_ar)
    
    # Train your network here. You'll probably need some weights and gradients!
    
    NN = NeuralNetwork()
    NN.val_out = bin_space
    
    epoch = 500 
    batch_size=25 
    lr = 0.5 
    for i in range(epoch):
        for batchx,batchy in batch_io(X,y,batch_size):
            grads = NN.computeGradients(batchx,batchy)
            params = NN.getParams()
            NN.setParams(params - lr*grads)
            
    return NN


def alv_vision(image, rgb, thresh):
    
        return (np.dot(image.reshape(-1, 3), rgb) > thresh).reshape(image.shape[0], image.shape[1])
    
def predict(NN, image_file):
    im_full = cv2.imread(image_file)
    processed_img = im_preprocess(im_full)
    yHat=NN.forward(processed_img)
    predicted_angle=NN.val_out[np.argmax(yHat)]
    return predicted_angle

def batch_io(X, y,size):
    for i in np.arange(0, X.shape[0], size):
        yield (X[i:i + size], y[i:i + size])
        
class NeuralNetwork(object):
    def __init__(self):        
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        self.inputLayerSize = 8460
        self.outputLayerSize = 90
        self.hiddenLayerSize = 45
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        val_out = 0
    
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
    

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
    
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))