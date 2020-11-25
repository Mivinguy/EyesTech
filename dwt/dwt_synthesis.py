###########

#

# DWT Synthesis V1.0

# 11/24/2020

###########

import numpy as np

def synthesis(bandLL, bandLH, bandHL, bandHH):
    """ Needs the original shape of the image to be restored  """

    # Synthesize bandL
    synthEvenBandL = bandLL - (bandLH[:-1,:] + bandLH[1:,:])/4
    synthOddBandL = bandLH[1:-1,:] + (synthEvenBandL[:-1,:]+synthEvenBandL[1:,:])/2
    synthBandL = np.zeros(((originalRow+1)%2 + originalRow, originalCol//2+1), dtype = np.float16)
    synthBandL[::2,:] = synthEvenBandL
    synthBandL[1::2,:] = synthOddBandL

    # Synthesize bandH
    synthEvenBandH = bandHL - (bandHH[:-1,:] + bandHH[1:,:])/4
    synthOddBandH = bandHH[1:-1,:] + (synthEvenBandH[:-1,:]+synthEvenBandH[1:,:])/2
    synthBandH = np.zeros(((originalRow+1)%2 + originalRow, originalCol//2+2), dtype = np.float16)
    synthBandH[::2,:] = synthEvenBandH
    synthBandH[1::2,:] = synthOddBandH

    # Restore image with bandL and bandH
    restoredImageEven = synthBandL - (synthBandH[:,:-1] + synthBandH[:,1:])/4
    restoredImageEven = restoredImageEven.astype('int8')
    restoredImageOdd = synthBandH[:,1:-1] + (restoredImageEven[:,:-1]+restoredImageEven[:,1:])/2
    restoredImageOdd = restoredImageOdd.astype('int8')

    restoredImage = np.zeros(np.shape(original), dtype = np.int8)

    # Need to think of cleaner implementation
    # By cases
    if(originalRow%2 == 0):
        restoredImage[:,1::2] = restoredImageOdd[:-1,:]
        if(originalCol%2 == 0):
            # Dims = (even, even)
            restoredImage[:,::2] = restoredImageEven[:-1,:-1]
        else:
            # Dims = (even, odd)
            restoredImage[:,::2] = restoredImageEven[:-1,:]
    else:
        restoredImage[:,1::2] = restoredImageOdd[:,:]
        if(originalCol%2 == 0):
            # Dims = (odd, even)
            restoredImage[:,::2] = restoredImageEven[:,:-1]
        else:
            # Dims = (odd, odd)
            restoredImage[:,::2] = restoredImageEven[:,:]

    restoredImage = restoredImage[:,:] + adjust[:,:]

    return restoredImage