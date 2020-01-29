import os
import shutil
from skimage import io
from skimage import img_as_ubyte
from skimage import filters
from skimage.util import invert
from matplotlib import pyplot as plt
from fuzzyTransform import fuzzyTransform
from skeleton2Graph import *
import skeletonization
import numpy as np
import scipy.ndimage.morphology as morph
from skimage.morphology import thin
import matplotlib.patches


def main():
    in_img_path = 'test_ml_comp_grey_framed'
    out_img_bdb_path = 'skeletonize_fuzzy_BDB'
    out_img_msdb_path = 'skeletonize_fuzzy_MSDB'
    out_img_flux_skel_path = 'skeletonize_flux'

    # delete out dirs and recreate
    shutil.rmtree(out_img_bdb_path, ignore_errors=True)
    os.makedirs(out_img_bdb_path)
    shutil.rmtree(out_img_msdb_path, ignore_errors=True)
    os.makedirs(out_img_msdb_path)
    shutil.rmtree(out_img_flux_skel_path, ignore_errors=True)
    os.makedirs(out_img_flux_skel_path)

    for i, img_path in enumerate(sorted(os.listdir(in_img_path))):
        print(img_path)
        img = io.imread(os.path.join(in_img_path, img_path),as_gray=True)
        img = invert(img)
        BW = mat2gray(img)

        thresh_img = filters.threshold_otsu(BW)
        BW = BW >= thresh_img
        BW = BW.astype(float)



        BW = 1 - BW
        M,N = BW.shape

        '''Calculating Euclidean Distance of the Binary Image'''
        D,IDX = morph.distance_transform_edt(BW,return_distances=True, return_indices=True)
        D = mat2gray(D)
        X,Y = np.meshgrid(range(N),range(M))
        delD_x = -(IDX[1,:,:] - X)
        delD_y = -(IDX[0,:,:] - Y)
        # normalize the derivatives
        delD_norm = np.sqrt(pow(delD_x,2) + pow(delD_y,2))
        with np.errstate(divide='ignore',invalid='ignore'):
            delD_xn   = delD_x / delD_norm
            delD_yn   = delD_y / delD_norm

        mir_delD_xn  = mirrorBW(delD_xn)
        mir_delD_yn  = mirrorBW(delD_yn)

        # f, (ax1, ax2,ax3) = plt.subplots(1,3,figsize=(10,30))
        # ax1.imshow(D)
        # ax1.set_title('Euclidean Distance Transform')
        # ax2.imshow(delD_x)
        # ax2.set_title('X direction of the gradient of EDT')
        # ax3.imshow(delD_y)
        # ax3.set_title('Y direction of the gradient of EDT')
        # plt.tight_layout()
        # plt.show()


        # #Calculate flux map
        # fluxMap = flux(mir_delD_xn,mir_delD_yn)
        # plt.imshow(np.nan_to_num(fluxMap))
        # plt.title('Flux Map')
        # plt.show()
        

        #Calculate flux map
        fluxMap = flux(mir_delD_xn,mir_delD_yn)
        # Calculate flux threshold
        print((np.nanmax(fluxMap) - np.nanmedian(fluxMap)))
        print(np.nanmedian(fluxMap))
        print(np.nanmax(fluxMap))
        print(np.nanmin(fluxMap))
        # fluxBWThreshold = (np.nanmax(fluxMap) - np.nanmedian(fluxMap)) * 0.30 + np.nanmedian(fluxMap)
        # fluxBWThreshold = (np.nanmax(fluxMap) - np.nanmean(fluxMap)) * 0.15 + np.nanmean(fluxMap)
        fluxBWThreshold = (np.nanmax(fluxMap) - np.nanmean(fluxMap)) * 0.6 + np.nanmean(fluxMap)
        print(fluxBWThreshold)
        with np.errstate(divide='ignore',invalid='ignore'):
            fluxThin = thin(fluxMap>fluxBWThreshold)

        # plt.imshow(np.nan_to_num(fluxMap>fluxBWThreshold))
        # plt.title('fluxMap>fluxBWThreshold Map')
        # plt.show()
        # # continue
        #
        # plt.imshow(np.nan_to_num(fluxThin))
        # plt.title('FluxThin Map')
        # plt.show()
        # # continue

        fluxLabeled,b = ndimage.label(fluxThin, np.array([[1,1,1], [1,1,1], [1,1,1]]))
        # print(fluxLabeled)
        # print(b)
        # fluxLabeled_1 = fluxLabeled == 1
        labels ,pixelSize = np.unique(fluxLabeled,return_counts=True)
        # print("labels: {}".format(labels))
        # print("pixelSize: {}".format(pixelSize))
        pixelSize_second = pixelSize
        pixelSize_second[np.argmax(pixelSize)] = 0
        skel_label = np.argmax(pixelSize_second)
        # print(skel_label)

        # Excluding the background
        pixelSize = pixelSize[labels != 0]
        labels = labels[labels != 0]
        # Calculating the size threshold and filter out small objects
        th = min(np.mean(pixelSize) + 3 * np.std(pixelSize), np.max(pixelSize))
        selectedObjects = labels[np.where(pixelSize >= th)]

        fluxTemp = np.zeros(fluxMap.shape)
        # fluxTemp[fluxLabeled == 1] = 1
        fluxTemp[fluxLabeled == skel_label] = 1
        # plt.imshow(fluxTemp,cmap='gray')
        # plt.title('Initial Skeleton with branches')
        # plt.tight_layout()
        # plt.show()



        binary_bdb = fluxTemp
        #thresh_bdb = filters.threshold_otsu(BDB)
        #binary_bdb = BDB >= thresh_bdb
        io.imsave(os.path.join(out_img_flux_skel_path, img_path), img_as_ubyte(binary_bdb))
        continue



        skeletonNew = np.zeros(fluxMap.shape)
        fluxTemp_fluxMap = fluxTemp*fluxMap
        adjacencyMatrix, edgeList, edgeProperties,edgeProperties2, verticesProperties, verticesProperties2, endPoints, branchPoints = skeleton2Graph(fluxTemp,fluxTemp*fluxMap)
        vertices = np.concatenate((endPoints, branchPoints))
        _,_,_, skeletonGraphPointsImg = findBranchPoints(fluxTemp,return_image=True)
        f, (ax1, ax2,ax3) = plt.subplots(1,3,figsize=(10,30))
        ax1.imshow(skeletonGraphPointsImg)
        ax1.set_title('Vertices of the Skeleton\'s graph')
        ax2.imshow(skeletonGraphPointsImg[50:100,125:170])
        ax2.set_title('Vertices, close look')
        ax3.imshow(graphDrawing(fluxTemp,edgeList,0.08))
        ax3.set_title('Edges of the Skeleton\'s graph')

        plt.tight_layout()
        plt.show()
        continue


        # skeletonNew, MSDB,BDB = fuzzyTransform(fluxTemp, vertices, edgeList, edgeProperties, verticesProperties, verticesProperties2, adjacencyMatrix, returnDB=True)
        # f, (ax1, ax2,ax3) = plt.subplots(1,3,figsize=(10,30))
        # ax1.imshow(MSDB,cmap ='gist_gray')
        # ax1.set_title('Main Skeleton Degree of Belief Map')
        # ax2.imshow(BDB,cmap ='gist_gray')
        # ax2.set_title('Branch Degree of Belief Map')
        # ax3.imshow(skeletonNew, cmap='gray')
        # ax3.set_title('Pruned Skeleton')
        # plt.tight_layout()
        # plt.show()



        # binary_bdb = BDB
        # #thresh_bdb = filters.threshold_otsu(BDB)
        # #binary_bdb = BDB >= thresh_bdb
        # io.imsave(os.path.join(out_img_bdb_path, img_path), img_as_ubyte(binary_bdb))



if __name__ == '__main__':
    main()
