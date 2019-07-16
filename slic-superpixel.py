# -*- coding: utf-8 -*-
"""
Created on Tuesday July 16 11:30:00 2019
E-mail = yuyafang1014@163.com
@author: Yuyafang
"""

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from osgeo import gdal
import cv2
import sys
import datetime
from skimage import segmentation
from skimage.future import graph
import numpy as np


def _weight_mean_color(graph, src, dst, n):
    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])


def daw_figure(numSegment, image, segment):
    fig = plt.figure("slic-superpixels -- %d segments" % (numSegment))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segment))
    plt.axis("off")
    return 0


def main():

    # ------Slic-Superpixel segmentation------

    if len(sys.argv) < 4:
        print("need more args")
        # TODO: print usage
        exit(-1)

    # Load the image(GDAL)
    dataset = gdal.Open(sys.argv[1])
    super_pixel_size = int(sys.argv[2])
    nc_sigma = int(sys.argv[3])
    scale = int(sys.argv[4])

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount

    if im_bands == 3:
        band1 = dataset.GetRasterBand(1)
        band2 = dataset.GetRasterBand(2)
        band3 = dataset.GetRasterBand(3)
        im_redBand = band1.ReadAsArray(0, 0, im_width, im_height)
        im_greenBand = band2.ReadAsArray(0, 0, im_width, im_height)
        im_blueBand = band3.ReadAsArray(0, 0, im_width, im_height)
        merge_img = cv2.merge([im_redBand, im_greenBand, im_blueBand])

    numSegments = int(im_width * im_height / super_pixel_size)

    start1 = datetime.datetime.now()

    # Apply SLIC and extract (approximately) the supplied number of segments
    segments = slic(merge_img, n_segments=numSegments, sigma=nc_sigma)
    print(segments)
    print(segments.shape)

    cv2.waitKey(0)

    # Show the output of SLIC
    end1 = datetime.datetime.now()
    print("The time spent in segmentation (seconds): ", (end1 - start1).seconds)
    daw_figure(numSegments, merge_img, segments)

    # Show the plots
    plt.show()

    # ------Hierarchical Region Merging Algorithm for Region Adjacent Graph (RAG)------
    start2 = datetime.datetime.now()

    # Compute the Region Adjacency Graph using mean colors
    g = graph.rag_mean_color(merge_img, segments)

    end21 = datetime.datetime.now()
    print("The time spent in mean colors (seconds): ", (end21 - start2).seconds)

    # Perform hierarchical merging of a RAG
    labels = graph.merge_hierarchical(segments, g, thresh=scale, rag_copy=False,
                                      in_place_merge=True,
                                      merge_func=merge_mean_color,
                                      weight_func=_weight_mean_color)

    # Return image with boundaries between labeled regions highlighted
    out = segmentation.mark_boundaries(segments, labels)

    # Show the figure
    end22 = datetime.datetime.now()
    fig = plt.figure("Region Merging -- %d scale" % (scale))
    ax = fig.add_subplot(1, 1, 1)
    print("The time spent on region merging (seconds): ", (end22 - start2).seconds)
    ax.imshow(out)
    plt.axis("off")
    plt.show()

    # ------Draw equalized histogram------
    start3 = datetime.datetime.now()
    arr = segments.flatten()

    # Show equalized histogram
    plt.figure("Equalized histogram")
    plt.hist(arr, bins=256, normed=1, edgecolor='None', facecolor='red')
    end3 = datetime.datetime.now()
    print("The time spent on equalized histogram (seconds): ", (end3 - start3).seconds)
    plt.show()


'''
usage:  python ./slic-superpixel.py input_image.tif super_pixel_size sigma scale
example: python ./slic-superpixel.py a.tif 3500 5 20

'''
if __name__ == '__main__':
    main()
