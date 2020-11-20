###########

#

# REPLAY MODULE

#

# (c) Copyright 2018-2020, Eyes Technology, Inc.

#

###########


import io
import socket
import struct
import cv2
import numpy as np
import time
## import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--transpose", help="replay file recorded in transposed format", action="store_true")
parser.add_argument("fileName", help="Recorded input file")
args = parser.parse_args()

if args.transpose:
    print("Playing from transposed format")
else:
    print("Playing from regular format")
## recFile = open(sys.argv[1], 'rb')
recFile = open(args.fileName, 'rb')
try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', recFile.read(struct.calcsize('<L')))[0]
        ##print 'Image length = ', image_len
        if not image_len:
            time.sleep(3)
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(recFile.read(image_len))
        # Rewind the stream, open it and do some processing on it
        image_stream.seek(0)
        data = np.fromstring(image_stream.getvalue(), dtype=np.uint8)
        if args.transpose:
            data.resize(318, 183)
            data = data.T
        else:
            data.resize(183, 318)             
        cv2.namedWindow("Display")
        cv2.imshow("Display", data)
        time.sleep(0.067)
        cv2.waitKey(10)

finally:

    recFile.close()

