# coding=utf-8
"""Implement Part Affinity Fields
:param centerA: int with shape (2,), centerA will pointed by centerB.
:param centerB: int with shape (2,), centerB will point to centerA.
:param accumulate_vec_map: one channel of paf.
:param count: store how many pafs overlaped in one coordinate of accumulate_vec_map.
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage
from mpl_toolkits.mplot3d import Axes3D


def putVecMaps(centerA, centerB, accumulate_vec_map, count, grid_y, grid_x, stride):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    thre = 2  # limb width
    # centerB = (centerB - 3.5) / stride
    # centerA = (centerA -3.5 ) / stride
    centerB = centerB  / stride
    centerA = centerA  / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if (norm == 0.0):
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0

    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[:, :, np.newaxis])
    accumulate_vec_map += vec_map
    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])
    count[mask == True] = 0

    return accumulate_vec_map, count

def putVecMasks(centerA, centerB, accumulate_vec_map, grid_y, grid_x, stride):
    
    
    start = stride / 2.0 - 0.5 #3.5
    y_range = [i for i in range(int(grid_y))] #46
    x_range = [i for i in range(int(grid_x))] #46
    xx, yy = np.meshgrid(x_range, y_range) 
    
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - centerA[0]) ** 2 + (yy - centerA[1]) ** 2
    # d21= (xx1 - center[0]) ** 2 + (yy1 - center[1]) ** 2
    # exponent1 = d21 / 2.0 / sigma / sigma
    exponent = d2 / 2.0 / 7 / 7
    mask = exponent <= 4.6052
    # mask1 = exponent1 <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_vec_map = np.where(accumulate_vec_map > cofid_map,accumulate_vec_map,cofid_map)

    d2 = (xx - centerB[0]) ** 2 + (yy - centerB[1]) ** 2
    # d21= (xx1 - center[0]) ** 2 + (yy1 - center[1]) ** 2
    # exponent1 = d21 / 2.0 / sigma / sigma
    exponent = d2 / 2.0 / 7 / 7
    mask = exponent <= 4.6052
    # mask1 = exponent1 <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_vec_map = np.where(accumulate_vec_map > cofid_map,accumulate_vec_map,cofid_map)


    



    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    thre = 1  # limb width
    centerB = (centerB - 3.5) / stride
    centerA = (centerA - 3.5 ) / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    # if (norm == 0.0):
    #     # print 'limb is too short, ignore it...'
    #     return accumulate_vec_map, count

    limb_vec_unit = limb_vec / norm
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)
    
    distancex = max_x- min_x
    distancey = max_y -min_y
    if distancey >= distancex:
        distancex += 1
        max_x += 1
    else:
        distancey += 1
    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    range_x1 = list(range(0, int(distancex), 1))
    range_y1 = list(range(0, int(distancey), 1))
    xx1,yy1 = np.meshgrid(range_x1,range_y1)
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])

    sigma = 1
    exponent = limb_width ** 2 / (2 * sigma * sigma)
    mask_paf_mask = exponent <= 4.6052
    mask_paf = np.exp(-exponent)
    mask_paf = np.multiply(mask_paf,mask_paf_mask)

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[yy, xx] = mask_paf[yy1,xx1]

    accumulate_vec_map = np.where(accumulate_vec_map > vec_map,accumulate_vec_map,vec_map)
     


    # mask = limb_width < thre  # mask is 2D

    # vec_map = np.copy(accumulate_vec_map) * 0.0
    # vec_map[yy, xx] = mask_paf[yy1,xx1]
    # vec_map = vec_map[np.newaxis,:,:]

    #vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    # vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    # mask = np.logical_or.reduce(
    #     (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    # accumulate_vec_map = np.multiply(
    #     accumulate_vec_map, count[:, :, np.newaxis])
    # accumulate_vec_map += vec_map
    # count[mask == True] += 1

    # mask = count == 0

    # count[mask == True] = 1

    # accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])
    # count[mask == True] = 0

    return accumulate_vec_map#accumulate_vec_map, count


if __name__ == "__main__":

    centerA = np.array((186,100))
    centerB = np.array((215,145))
    paf_mask = np.zeros([38,46,46])
    paf_check = np.zeros([46,46,38])

    
    i = 0
    
     
    
    count = np.zeros((46,46),dtype=float)
    paf_mask[i,:,:] = putVecMasks(centerA,centerB,paf_mask[i,:,:],46,46,8)

    paf_check[:,:,i+1:i+3],count= putVecMaps(centerA,centerB,paf_check[:,:,i+1:i+3],count,46,46,8)
    # center = [108,108]
    # heatmap[i,:,:],heatmap[i+1,:,:] = putGaussianMaps(center,heatmap[i,:,:],heatmap[i+1,:,:],7,46,46,8)
   
    #x,y = np.meshgrid(x,y)


    a = plt.figure()
    ax =Axes3D(a)
    x = np.arange(0,46,1)
    y = np.arange(0,46,1)
    x,y = np.meshgrid(x,y)
    paf_mask[i,45,45] = 5
    def fun(x,y):
        return paf_mask[i,x,y]
    z = fun(x,y)
    ax.plot_surface(x,y,z,cmap="rainbow")
    #plt.draw()

    # b = a.add_subplot(221)
    # plt.imshow(paf_mask[i,:,:])
    # plt.colorbar()
    # c = a.add_subplot(222)
    # plt.imshow(paf_check[:,:,i+1])
    # plt.colorbar()
    # d = a.add_subplot(223)
    # plt.imshow(paf_check[:,:,i+2])
    # plt.colorbar()
    # e = a.add_subplot(224)
    # plt.imshow(np.multiply(paf_check[:,:,i+1],paf_mask[i,:,:]))
    # plt.colorbar()

    #c.imshow(heatmap[i+1,:,:])
    plt.show()
