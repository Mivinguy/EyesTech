###########

#

# DWT

#

###########

import struct
import time
import numpy as np
from padding import padArray
import io


fileNameR = 'singleFrame.txt'
inFile = open(fileNameR, 'rb')

image_len = struct.unpack('<L', inFile.read(struct.calcsize('<L')))[0]
if not image_len:
    time.sleep(3)
    print('Image length is 0 ')
image_stream = io.BytesIO()
image_stream.write(inFile.read(image_len))
image_stream.seek(0)
original = np.frombuffer(image_stream.getvalue(), dtype=np.uint8).reshape(318,183) # reshaped the test image, it is originally of dimension (58194,)
adjust = np.zeros(np.shape(original), dtype = np.uint8)
adjust[:,:] = 128
data1 = original[:,:] - adjust[:,:]
# Pad the data
paddedData1 = padArray(data1, np.shape(data1))


#####     Only works for input dimensions of even rows, odd columns || Ex. (318,183)     #####

""" Make false, if you want to run the code with the test image || Make true to run with randomized array """
testWithRandom = True

if(testWithRandom):
    original = np.random.randint(255, size=(18, 15))
    adjust = np.zeros(np.shape(original), dtype = np.uint8)
    adjust[:,:] = 128
    data1 = original[:,:] - adjust[:,:]
    paddedData1 = padArray(data1, np.shape(data1))

    # Analysis
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

    # Synthesize bandL
    synthEvenBandL = bandLL - (bandLH[:-1,:] + bandLH[1:,:])/4
    synthOddBandL = bandLH[1:-1,:] + (synthEvenBandL[:-1,:]+synthEvenBandL[1:,:])/2
    synthBandL = np.zeros((np.shape(data1)[0]+1, np.shape(data1)[1]//2+1), dtype = np.float16)  # This needs to change to work for all cases
    synthBandL[::2,:] = synthEvenBandL
    synthBandL[1::2,:] = synthOddBandL

    # Synthesize bandH
    synthEvenBandH = bandHL - (bandHH[:-1,:] + bandHH[1:,:])/4
    synthOddBandH = bandHH[1:-1,:] + (synthEvenBandH[:-1,:]+synthEvenBandH[1:,:])/2
    synthBandH = np.zeros((np.shape(data1)[0]+1, np.shape(data1)[1]//2+2), dtype = np.float16)  # This needs to change to work for all cases
    synthBandH[::2,:] = synthEvenBandH
    synthBandH[1::2,:] = synthOddBandH

    # Restore image with bandL and bandH
    restoredImageEven = synthBandL - (synthBandH[:,:-1] + synthBandH[:,1:])/4
    restoredImageEven = restoredImageEven.astype('int8')
    restoredImageOdd = synthBandH[:,1:-1] + (restoredImageEven[:,:-1]+restoredImageEven[:,1:])/2
    restoredImageOdd = restoredImageOdd.astype('int8')

    restoredImage = np.zeros(np.shape(data1), dtype = np.int8)
    restoredImage[:,::2] = restoredImageEven[:-1,:]                                             # This needs to change to work for all cases
    restoredImage[:,1::2] = restoredImageOdd[:-1,:]                                             # This needs to change to work for all cases
    restoredImage = restoredImage[:,:] + adjust[:,:]                                            

    print("Original image: \n", original)
    print("restoredImage : \n", restoredImage)    
    np.testing.assert_array_equal(restoredImage, original)

else:
    # Analysis
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

    # Synthesize bandL
    synthEvenBandL = bandLL - (bandLH[:-1,:] + bandLH[1:,:])/4
    synthOddBandL = bandLH[1:-1,:] + (synthEvenBandL[:-1,:]+synthEvenBandL[1:,:])/2
    synthBandL = np.zeros((np.shape(data1)[0]+1, np.shape(data1)[1]//2+1), dtype = np.float16)  # This needs to change to work for all cases
    synthBandL[::2,:] = synthEvenBandL
    synthBandL[1::2,:] = synthOddBandL

    # Synthesize bandH
    synthEvenBandH = bandHL - (bandHH[:-1,:] + bandHH[1:,:])/4
    synthOddBandH = bandHH[1:-1,:] + (synthEvenBandH[:-1,:]+synthEvenBandH[1:,:])/2
    synthBandH = np.zeros((np.shape(data1)[0]+1, np.shape(data1)[1]//2+2), dtype = np.float16)  # This needs to change to work for all cases
    synthBandH[::2,:] = synthEvenBandH
    synthBandH[1::2,:] = synthOddBandH

    # Restore image with bandL and bandH
    restoredImageEven = synthBandL - (synthBandH[:,:-1] + synthBandH[:,1:])/4
    restoredImageEven = restoredImageEven.astype('int8')
    restoredImageOdd = synthBandH[:,1:-1] + (restoredImageEven[:,:-1]+restoredImageEven[:,1:])/2
    restoredImageOdd = restoredImageOdd.astype('int8')

    restoredImage = np.zeros(np.shape(data1), dtype = np.int8)
    restoredImage[:,::2] = restoredImageEven[:-1,:]                                             # This needs to change to work for all cases
    restoredImage[:,1::2] = restoredImageOdd[:-1,:]
    restoredImage = restoredImage[:,:] + adjust[:,:]                                             # This needs to change to work for all cases

    print("Original image: \n", original)
    print("restoredImage : \n", restoredImage)    
    np.testing.assert_array_equal(restoredImage, original)