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
#original = np.random.randint(255, size=(318,183))
lowestBandLL = decomposition(original, 1, num_of_passes)
restoredImage = synthesis(lowestBandLL, num_of_passes)

print("Restored from lowest level bandLL: \n", restoredImage)
print('\n')
print("Original: \n", original)

np.testing.assert_array_equal(original, restoredImage)

"""
Turned everything to int16, solved all problems
info below is for trying to make everything in int8

UPdate: Only the odd rows are incorrect, even rows are perfect
        some numbers are going over int8 range(-128,127) when 
        doing more than 1 decomposition level 

        might be a bandLL problem as noted by rasha:
            All the analysis/synthesis ops should eventually be done 
            completely in integers; they start off as 0-centered int8, 
            which might suffice for all the H bands (actual values prior to compression), 
            though the LL bands (both intermediates and the final one) will gradually grow 
            in dynamic range and so require int16 (see (c) below; it’s easy to see why and 
            you likely already have in practice).  
            The rounding (which is what they occasionally refer to as “minor non-linearities”) 
            is always down (note those brackets in the formulas) and if done systematically 
            and consistently should continue to yield perfect round-trip reconstruction, as currently

Need to modify huffman encoding to only deal with ints (currently dealing with floats)
"""

