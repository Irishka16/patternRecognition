import numpy as np
import random

def sp_noise_gray(image, prob=0.03):
    '''
    Add salt and pepper noise to a gray image [0,255]
    
    image: Numpy 2D array
    prob: Probability of the noise
    returns: Numpy 2D array
    '''    
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                image[i,j] = 0
            elif rdn > thres:
                image[i,j] = 255
    return image

def sp_noise_color(image, prob=0.03, white=[255,255,255], black=[0,0,0]):
    '''
    Add salt and pepper noise to a color image
    
    image: Numpy 2D array
    prob: Probability of the noise
    returns: Numpy 2D array
    '''    
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                image[i,j,:] = black
            elif rdn > thres:
                image[i,j,:] = white
    return image

def norm_noise_gray(image, mean=0, var=0.1, a=0.5):
    '''
    Add gaussian noise to gray image 
    
    image: Numpy 2D array
    mean: scalar
    vat: scalar
    returns: Numpy 2D array
    '''    
    sigma = var**0.5
    
    row,col= image.shape[:2]
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = a*image + (1-a)*gauss

    noisy = noisy-np.min(noisy)
    noisy = 255*(noisy/np.max(noisy))
    
    return noisy.astype(np.uint8)

def norm_noise_color(image, mean=0, var=0.1, a=0.5):
    '''
    Add gaussian noise to color image 
    
    image: Numpy 2D array
    mean: scalar - mean
    var: scalar - variance
    a: scalar [0-1] - alpha blend
    returns: Numpy 2D array
    '''    
    sigma = var**0.5
    
    row,col,ch= image.shape[:3]
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = a*image + (1-a)*gauss

    noisy = noisy-np.min(noisy)
    noisy = 255*(noisy/np.max(noisy))
    
    return noisy.astype(np.uint8)