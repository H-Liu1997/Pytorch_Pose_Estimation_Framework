# coding=utf-8

import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage


"""Implement the generate of every channel of ground truth heatmap.
:param centerA: int with shape (2,), every coordinate of person's keypoint.
:param accumulate_confid_map: one channel of heatmap, which is accumulated, 
       np.log(100) is the max value of heatmap.
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""


def putGaussianMaps(center, accumulate_confid_map,sigma, grid_y, grid_x, stride):

    start = stride / 2.0 - 0.5 #3.5
    y_range = [i for i in range(int(grid_y))] #46
    x_range = [i for i in range(int(grid_x))] #46
    xx, yy = np.meshgrid(x_range, y_range) 
    # xx1, yy1 = np.meshgrid(x_range, y_range)
    # xx1 = xx1 * 8 + 3.5
    # yy1 = yy1 * 8 + 3.5
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    # d21= (xx1 - center[0]) ** 2 + (yy1 - center[1]) ** 2
    # exponent1 = d21 / 2.0 / sigma / sigma
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    # mask1 = exponent1 <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map = np.where(accumulate_confid_map > cofid_map,accumulate_confid_map,cofid_map)
    #accumulate_confid_map += cofid_map
    #accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    # cofid_map1 = np.exp(-exponent1)
    # cofid_map1 = np.multiply(mask1, cofid_map1)
    # accumulate_confid_map1 = np.where(accumulate_confid_map1 > cofid_map1,accumulate_confid_map1,cofid_map1)
    # accumulate_confid_map1[accumulate_confid_map1 > 1.0] = 1.0
    
    return accumulate_confid_map #accumulate_confid_map1

if __name__ == "__main__":

    center = [100,100]
    heatmap = np.zeros([19,60,60])
    i = 0
    heatmap[i,:,:] = putGaussianMaps(center,heatmap[i,:,:],7,60,60,8)
    center = [108,108]
    heatmap[i,:,:]= putGaussianMaps(center,heatmap[i,:,:],7,60,60,8)

    heatmap_ = cv2.resize(heatmap[i,:,:],(120,120),interpolation=cv2.INTER_CUBIC)
    
    a = plt.figure()
    b = a.add_subplot(121)
    b.imshow(heatmap[i,:,:])
    c = a.add_subplot(122)
    c.imshow(heatmap_)
    plt.show()
    
