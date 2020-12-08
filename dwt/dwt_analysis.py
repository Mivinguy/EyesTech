###########

#

# DWT Analysis V1.0

# 11/24/2020

# Accepts an image to be analyzed and level of decomposition desired
#   returns lowest level bandLL to be stored in the double buffer
#   compresses H bands with Huffman coding and packages them out

###########

import numpy as np
from padding import padArray

def decomposition(image, levels):
    originalRow, originalCol = np.shape(image)
    adjust = np.zeros(np.shape(image), dtype = np.uint8)
    adjust[:,:] = 128
    data1 = image[:,:] - adjust[:,:]
    # Pad the data
    paddedData1 = padArray(data1, np.shape(data1))

    # Horizontal pass
    even = paddedData1[:,::2]
    odd = paddedData1[:,1::2]
    bandH = odd - (even[:,:-1]+even[:,1:])/2
    bandL = even[:,1:-1] + (bandH[:,:-1] + bandH[:,1:])/4

    # Vertical pass
    even = bandL[::2,:]
    odd = bandL[1::2,:]
    bandLH = odd - (even[:-1,:]+even[1:,:])/2
    bandLL = even[1:-1,:] + (bandLH[:-1,:] + bandLH[1:,:])/4

    even = bandH[::2,:]
    odd = bandH[1::2,:]
    bandHH = odd - (even[:-1,:]+even[1:,:])/2
    bandHL = even[1:-1,:] + (bandHH[:-1,:] + bandHH[1:,:])/4

    print(np.shape(bandLL), " current level: ", levels)

    """ Compress and package the 3 H bands here """

    if levels <= 1:
        return bandLL
    else:
        return decomposition(bandLL, levels - 1)
