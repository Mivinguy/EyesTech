###########

#

# DWT test

# 12/15/2020

# Test file to verify analysis/synthesis

###########

import struct
import time
import numpy as np
from dwt_analysis import decomposition
from dwt_synthesis import synthesis
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

num_of_passes = 4

original = np.frombuffer(image_stream.getvalue(), dtype=np.uint8).reshape(318,183) # reshaped the test image, it is originally of dimension (58194,)

lowestBandLL = decomposition(original, 1, num_of_passes)
restoredImage = synthesis(lowestBandLL, num_of_passes)

print("Restored from lowest level bandLL: \n", restoredImage)
print('\n')
print("Original: \n", original)

np.testing.assert_array_equal(original, restoredImage)

