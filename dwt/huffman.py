###########

#

# Huffman coding V1.0

# 11/24/2020

# To compress H bands 

# To Do:
#   1. write decompression function
#       - Read in full binary string 
#       - Map binary strings to image pixel values
#       - Build original image

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

    def __repr__(self):
        # Return string representation
        return "Node {}, left {}, right {} || ".format(self.val, self.left, self.right)

class huffmanCoding:
    def __init__(self, image):
        self.image = image
        self.heap = []
        self.heapCopy = []
        self.codeBook = {}

    def frequency(self, image):
        # Count frequency of each pixel value
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
        root = heapq.heappop(self.heapCopy)
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
        # Convert image to a string of codes then write to binary file
        codeStr = ""
        for row in range(np.shape(image)[0]):
            for col in range(np.shape(image)[1]):
                codeStr += self.codeBook[(image[row][col])]
        stream.write(self.toBytes(codeStr))

    def toBytes(self, data):
        # Converts binary string to bytes
        b = bytearray()
        for i in range(0, len(data), 8):
            b.append(int(data[i:i+8], 2))
        return bytes(b)

    def encodeDims(self, stream):
        # Encode original image dimensions as padded 16 bit values (2 bytes each)
        rows = np.shape(self.image)[0]
        binRow = f'{rows:016b}'
        stream.write(self.toBytes(binRow))

        cols = np.shape(self.image)[1]
        binCol = f'{cols:016b}'
        stream.write(self.toBytes(binCol))

    def encodeTree(self, stream):
        root = heapq.heappop(self.heap)

        # Make a copy of the original heap to build code book
        heapq.heappush(self.heapCopy, root)
        
        self.encodeTreeHelper(root, stream)

    def encodeTreeHelper(self, node, stream):
        if (node.left == None and node.right == None):
            # Leaf node, write 1, followed by value 
            stream.write((struct.pack('b', 1)))

            # Value is 4 bytes
            data = struct.pack('f', node.val)
            #print(struct.unpack('f',data)[0])
            stream.write(data)

        else:
            # Parent node, write 0, continue down
            stream.write((struct.pack('b', 0)))
            self.encodeTreeHelper(node.left, stream)
            self.encodeTreeHelper(node.right, stream)

    def compress(self):
        fileNameW = 'compressedFrame.bin'
        outFile = open(fileNameW, 'wb')
        freq = self.frequency(self.image)
        self.createHeap(freq)
        self.buildTree()

        # First 4 bytes will always be original dimensions
        self.encodeDims(outFile) 

        # Write encoded tree to file
        self.encodeTree(outFile)

        self.buildCodebook()

        # Then write the compressed pixel values
        self.encodeImage(self.image, outFile)

    def decompress(self):
        fileNameW = 'compressedFrame.bin'
        inFile = open(fileNameW, 'rb')

        # Read first 4 bytes to get dimensions
        rowsBytes = inFile.read(2)
        originalRows = int.from_bytes(rowsBytes, "big")
        colsBytes = inFile.read(2)
        originalCols = int.from_bytes(colsBytes, "big")

        # Begin reconstructing huffman tree
        huffmanTree = self.decompressTree(inFile)
        
        # Read in string of codes


        # Use string of codes to rebuild image


    
    def decompressTree(self, stream):
        byte = stream.read(1)
        if struct.unpack('b',byte)[0] == 1:
            packedValue = stream.read(4)
            value = struct.unpack('f',packedValue)[0]
            return value
        else:
            left = self.decompressTree(stream)
            right = self.decompressTree(stream)
            return (left, right)



""" Testing """

originalHH = np.load('outfileHH.npy')
originalHL = np.load('outfileHL.npy')
originalLH = np.load('outfileLH.npy')
huffmanCoding(originalHH).compress()
huffmanCoding(originalHH).decompress()

