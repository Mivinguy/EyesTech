###########

#

# Huffman coding V1.0

# 11/24/2020

# To compress H bands 

###########

import struct
import time
import numpy as np
from dwt_analysis import decomposition
import io
import heapq


class Node:
	def __init__(self, val, freq):
		self.val = val
		self.freq = freq
		self.left = None
		self.right = None

	def __lt__(self, other):
		return self.freq < other.freq

class huffmanCoding:
    def __init__(self, image):
        self.image = image
        self.heap = []
        self.codeBook = {}
        self.bitBuffer = []

    def frequency(self, image):
        freq = {}
        for row in range(np.shape(image)[0]):
            for col in range(np.shape(image)[1]):
                if image[row][col] in freq:
                    freq[image[row][col]] += 1
                else:
                    freq[image[row][col]] = 1
        return freq

    def createHeap(self, frequency):
        for val in frequency:
            node = Node(val, frequency[val])
            heapq.heappush(self.heap, node)
    
    def buildTree(self):
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            parent = Node(None, node1.freq + node2.freq)
            parent.left = node1
            parent.right = node2

            heapq.heappush(self.heap, parent)

    def buildCodebook(self):
        root = heapq.heappop(self.heap)
        currentCode = ""
        self.buildCodebookHelper(root, currentCode)

    def buildCodebookHelper(self, root, currentCode):
        if(root == None):
            return
        if(root.val != None):
            self.codeBook[root.val] = currentCode
            return

        self.buildCodebookHelper(root.left, currentCode + "0")
        self.buildCodebookHelper(root.right, currentCode + "1")

    def encodeImage(self, image, stream):
        for row in range(np.shape(image)[0]):
            for col in range(np.shape(image)[1]):
                currentCode = self.codeBook[(image[row][col])]
                self.bitsToBuffer(currentCode, stream)

    def bitsToBuffer(self, bitStr, stream):
        if bitStr is not None:
            for bit in range(len(bitStr)):
                self.bitBuffer.append(bitStr[bit])
                if len(self.bitBuffer) >= 8:
                    self.bufferToFile(stream)
    
    def bufferToFile(self, stream):
        byte = self.bitBuffer[:8]
        self.bitBuffer = self.bitBuffer[8:]
        str = ""
        intValue = int(str.join(byte), 2)
        stream.write(bytes(intValue))

    def compress(self):
        fileNameW = 'compressedFrame.bin'
        outFile = open(fileNameW, 'wb')
        freq = self.frequency(self.image)
        self.createHeap(freq)
        self.buildTree()
        self.buildCodebook()
        print(len(self.codeBook))
        encodedImage = self.encodeImage(self.image, outFile)
        self.bitsToBuffer(encodedImage, outFile)
        #fileNameW = 'compressedFrame.bin'
        #outFile = open(fileNameW, 'wb')
        #outFile.write(bytes(b))
        #print(self.codeBook)

""" Testing """

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
huffmanCoding(original).compress()
print('input image size: ', original.size)

