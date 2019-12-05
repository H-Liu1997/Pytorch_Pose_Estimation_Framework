#!/usr/bin/env python

import numpy as np
from math import sqrt, isnan

from .py_rmpe_config_offset import RmpeGlobalConfig, TransformationParams

class Heatmapper:

    def __init__(self, sigma=TransformationParams.sigma, thre=TransformationParams.paf_thre):

        self.double_sigma2 = 2 * sigma * sigma
        self.thre = thre

        # cached common parameters which same for all iterations and all pictures
        stride = RmpeGlobalConfig.stride
        self.stride = RmpeGlobalConfig.stride
        width = RmpeGlobalConfig.width//stride
        height = RmpeGlobalConfig.height//stride
        

        # this is coordinates of centers of bigger grid
        self.grid_x = np.arange(width)*stride + stride/2-0.5
        self.grid_y = np.arange(height)*stride + stride/2-0.5

        self.Y, self.X = np.mgrid[0:RmpeGlobalConfig.height:stride,0:RmpeGlobalConfig.width:stride]

        # TODO: check it again
        # basically we should use center of grid, but in this place classic implementation uses left-top point.
        # self.X = self.X + stride / 2 - 0.5
        # self.Y = self.Y + stride / 2 - 0.5

    def create_heatmaps(self, joints, mask):

        heatmaps = np.zeros(RmpeGlobalConfig.parts_shape, dtype=np.float)

        self.put_joints(heatmaps, joints)
        sl = slice(RmpeGlobalConfig.heat_start, RmpeGlobalConfig.heat_start + RmpeGlobalConfig.heat_layers) 
        heatmaps[RmpeGlobalConfig.bkg_start] = 1. - np.amax(heatmaps[sl,:,:], axis=0)

        slx = slice(RmpeGlobalConfig.offset_start, RmpeGlobalConfig.offset_start + RmpeGlobalConfig.num_offset,2)
        sly = slice(RmpeGlobalConfig.offset_start+1, RmpeGlobalConfig.offset_start+1 + RmpeGlobalConfig.num_offset,2)
        heatmaps[RmpeGlobalConfig.offset_bkg_start] = 1. - np.amax(heatmaps[slx,:,:], axis=0)
        heatmaps[RmpeGlobalConfig.offset_bkg_start+1] = 1. - np.amax(heatmaps[sly,:,:], axis=0)
        self.put_limbs(heatmaps, joints)

        heatmaps *= mask

        return heatmaps

    def create_heatmaps_test(self, joints):

        heatmaps = np.zeros(RmpeGlobalConfig.parts_shape, dtype=np.float)

        self.put_joints_test(heatmaps, joints)
        sl = slice(RmpeGlobalConfig.heat_start, RmpeGlobalConfig.heat_start + RmpeGlobalConfig.heat_layers) 
        heatmaps[RmpeGlobalConfig.bkg_start] = 1. - np.amax(heatmaps[sl,:,:], axis=0)

        slx = slice(RmpeGlobalConfig.offset_start, RmpeGlobalConfig.offset_start + RmpeGlobalConfig.num_offset,2)
        sly = slice(RmpeGlobalConfig.offset_start+1, RmpeGlobalConfig.offset_start+1 + RmpeGlobalConfig.num_offset,2)
        heatmaps[RmpeGlobalConfig.offset_bkg_start] = 1. - np.amax(heatmaps[slx,:,:], axis=0)
        heatmaps[RmpeGlobalConfig.offset_bkg_start+1] = 1. - np.amax(heatmaps[sly,:,:], axis=0)
        #self.put_limbs(heatmaps, joints)

        return heatmaps


    def put_offset_maps(self, heatmaps, layer, joints):
        for i in range(joints.shape[0]):
        #for i in range(1):
            dis_x = self.grid_x-joints[i,0]
            dis_y = self.grid_y-joints[i,1]
            # dis_x = self.grid_x-100
            # dis_y = self.grid_y-181
            # print(np.min(np.abs(dis_x)))
            offset_x_index = np.where(np.abs(dis_x) == np.min(np.abs(dis_x)))
            offset_y_index = np.where(np.abs(dis_y) == np.min(np.abs(dis_y)))
            #print(offset_x_index)
            #print(offset_y_index)
            offset_x = dis_x[offset_x_index] / self.stride + 0.5 
            offset_y = dis_y[offset_y_index] / self.stride + 0.5
            #TODO: check the position
            heatmaps[RmpeGlobalConfig.offset_start + 2 * layer, offset_x_index, offset_y_index] = offset_x
            heatmaps[RmpeGlobalConfig.offset_start + 2 * layer + 1, offset_x_index, offset_y_index] = offset_y


    def put_gaussian_maps(self, heatmaps, layer, joints):
        # actually exp(a+b) = exp(a)*exp(b), lets use it calculating 2d exponent, it could just be calculated by

        for i in range(joints.shape[0]):
        #for i in range(1):    
            exp_x = np.exp(-(self.grid_x-joints[i,0])**2/self.double_sigma2)
            exp_y = np.exp(-(self.grid_y-joints[i,1])**2/self.double_sigma2)
            # exp_x = np.exp(-(self.grid_x-100)**2/self.double_sigma2)
            # exp_y = np.exp(-(self.grid_y-112)**2/self.double_sigma2)

            exp = np.outer(exp_y, exp_x)
            # note this is correct way of combination - min(sum(...),1.0) as was in C++ code is incorrect
            # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/118
            heatmaps[RmpeGlobalConfig.heat_start + layer, :, :] = np.maximum(heatmaps[RmpeGlobalConfig.heat_start + layer, :, :], exp)

    def put_joints(self, heatmaps, joints):
        for i in range(RmpeGlobalConfig.num_parts):
            visible = joints[:,i,2] < 2
            self.put_gaussian_maps(heatmaps, i, joints[visible, i, 0:2])
            self.put_offset_maps(heatmaps, i, joints[visible, i, 0:2])
    
    def put_joints_test(self, heatmaps, joints):
            self.put_gaussian_maps(heatmaps, 0, joints[:, 0:2])
            self.put_offset_maps(heatmaps, 0, joints[:, 0:2])


    def putVecMasks(self, centerA, centerB, accumulate_vec_map, stride):
    
        grid_y, grid_x = self.grid_y, self.grid_x
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

        sigma = 0.7
        exponent = limb_width ** 2 / (2 * sigma * sigma)
        mask_paf_mask = exponent <= 4.6052
        mask_paf = np.exp(-exponent)
        mask_paf = np.multiply(mask_paf,mask_paf_mask)

        vec_map = np.copy(accumulate_vec_map) * 0.0
        vec_map[yy, xx] = mask_paf[yy1,xx1]

        accumulate_vec_map = np.where(accumulate_vec_map > vec_map,accumulate_vec_map,vec_map)
        return accumulate_vec_map


    def put_vector_maps(self, heatmaps, layerX, layerY, joint_from, joint_to):

        count = np.zeros(heatmaps.shape[1:], dtype=np.int)

        for i in range(joint_from.shape[0]):
            (x1, y1) = joint_from[i]
            (x2, y2) = joint_to[i]

            dx = x2-x1
            dy = y2-y1
            dnorm = sqrt(dx*dx + dy*dy)

            if dnorm==0:  # we get nan here sometimes, it's kills NN
                # TODO: handle it better. probably we should add zero paf, centered paf, or skip this completely
                #print("Parts are too close to each other. Length is zero. Skipping")
                continue

            dx = dx / dnorm
            dy = dy / dnorm

            assert not isnan(dx) and not isnan(dy), "dnorm is zero, wtf"

            min_sx, max_sx = (x1, x2) if x1 < x2 else (x2, x1)
            min_sy, max_sy = (y1, y2) if y1 < y2 else (y2, y1)

            min_sx = int(round((min_sx - self.thre) / RmpeGlobalConfig.stride))
            min_sy = int(round((min_sy - self.thre) / RmpeGlobalConfig.stride))
            max_sx = int(round((max_sx + self.thre) / RmpeGlobalConfig.stride))
            max_sy = int(round((max_sy + self.thre) / RmpeGlobalConfig.stride))

            # check PAF off screen. do not really need to do it with max>grid size
            if max_sy < 0:
                continue

            if max_sx < 0:
                continue

            if min_sx < 0:
                min_sx = 0

            if min_sy < 0:
                min_sy = 0

            #TODO: check it again
            slice_x = slice(min_sx, max_sx) # + 1     this mask is not only speed up but crops paf really. This copied from original code
            slice_y = slice(min_sy, max_sy) # + 1     int g_y = min_y; g_y < max_y; g_y++ -- note strict <

            dist = distances(self.X[slice_y,slice_x], self.Y[slice_y,slice_x], x1, y1, x2, y2)
            dist = dist <= self.thre

            # TODO: averaging by pafs mentioned in the paper but never worked in C++ augmentation code
            heatmaps[layerX, slice_y, slice_x][dist] = (dist * dx)[dist]  # += dist * dx
            heatmaps[layerY, slice_y, slice_x][dist] = (dist * dy)[dist] # += dist * dy
            count[slice_y, slice_x][dist] += 1

        # TODO: averaging by pafs mentioned in the paper but never worked in C++ augmentation code
        # heatmaps[layerX, :, :][count > 0] /= count[count > 0]
        # heatmaps[layerY, :, :][count > 0] /= count[count > 0]


    def put_limbs(self, heatmaps, joints):

        for (i,(fr,to)) in enumerate(RmpeGlobalConfig.limbs_conn):


            visible_from = joints[:,fr,2] < 2
            visible_to = joints[:,to, 2] < 2
            visible = visible_from & visible_to

            layerX, layerY = (RmpeGlobalConfig.paf_start + i*2, RmpeGlobalConfig.paf_start + i*2 + 1)
            self.put_vector_maps(heatmaps, layerX, layerY, joints[visible, fr, 0:2], joints[visible, to, 0:2])
            #heatmaps[RmpeGlobalConfig.paf_mask_start+i] = self.putVecMasks(joints[visible, fr, 0:2], 
                                    #joints[visible, to, 0:2],heatmaps[RmpeGlobalConfig.paf_mask_start+i],8)



#parallel calculation distance from any number of points of arbitrary shape(X, Y), to line defined by segment (x1,y1) -> (x2, y2)

def distances(X, Y, x1, y1, x2, y2):

    # classic formula is:
    # d = (x2-x1)*(y1-y)-(x1-x)*(y2-y1)/sqrt((x2-x1)**2 + (y2-y1)**2)

    xD = (x2-x1)
    yD = (y2-y1)
    norm2 = sqrt(xD**2 + yD**2)
    dist = xD*(y1-Y)-(x1-X)*yD
    dist /= norm2

    return np.abs(dist)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hm = Heatmapper()
    Joints = np.array([[150,215,1]])
    heatmap_test = hm.create_heatmaps_test(Joints)
    fig = plt.figure()
    a = fig.add_subplot(2,2,1)
    a.set_title('offset_x')
    plt.imshow(heatmap_test[57])
    plt.colorbar()
    b = fig.add_subplot(2,2,2)
    b.set_title('offset_y')
    plt.imshow(heatmap_test[58])
    plt.colorbar()
    c = fig.add_subplot(2,2,3)
    c.set_title('x_background')
    plt.imshow(heatmap_test[-2])
    plt.colorbar()
    d = fig.add_subplot(2,2,4)
    d.set_title('y_background')
    plt.imshow(heatmap_test[-1])
    plt.colorbar()
    plt.show()
