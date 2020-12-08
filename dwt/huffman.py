###########

#

# Huffman coding V1.1

# 12/04/2020

# To compress H bands 

# Successfully compresses bandHH of singleFrame.txt to size     19KB       (numpy.save = 118KB)
#              compresses bandHL of singleFrame.txt to size     38KB       (numpy.save = 117KB)
#              compresses bandLH of singleFrame.txt to size     30KB       (numpy.save = 116KB)
#   and decompresses and restores an identical image

# To Do:
#   Compress multiple bands (HH + HL + LH) into 1 file and restore all of them
#   Decide on file naming conventions

#

###########

import struct
import time
import numpy as np
import io
import heapq


class Node:
    def __init__(self, val, freq):
        self.val = val
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        # Heap requires comparison method
        return self.freq < other.freq

    def __repr__(self):
        # Return string representation
        return "Node {}, left {}, right {} || ".format(self.val, self.left, self.right)

class huffmanCoding:
    def __init__(self, bandHH, bandHL, bandLH):
        self.bandHH = bandHH
        self.bandHL = bandHL
        self.bandLH = bandLH
        self.heap = []
        self.codeBook = {}
        self.binaryStr = ""
        self.buffer = ""

    def frequency(self, band):
        # Count frequency of each pixel value
        freq = {}
        for row in range(np.shape(band)[0]):
            for col in range(np.shape(band)[1]):
                if band[row][col] in freq:
                    freq[band[row][col]] += 1
                else:
                    freq[band[row][col]] = 1
        return freq

    def createHeap(self, frequency):
        # Using heap for ease of removing lowest freq nodes
        for val in frequency:
            node = Node(val, frequency[val])
            heapq.heappush(self.heap, node)
    
    def buildTree(self):
        # Use heap to create Huffman tree
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            parent = Node(None, node1.freq + node2.freq)
            parent.left = node1
            parent.right = node2

            heapq.heappush(self.heap, parent)

    def buildCodebook(self):
        # Build compressed codes from tree
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

    def toBuffer(self, data, stream):
        # Converts binary string to bytes, stores into buffer
        self.buffer += data
        if len(self.buffer) >= 8:
            self.toFile(stream)
        else:
            return
    
    def toFile(self, stream):
        # From buffer, write to file 1 byte at a time
        b = bytearray()
        b.append(int(self.buffer[:8], 2))
        self.buffer = self.buffer[8:]
        stream.write(bytes(b))
        if len(self.buffer) >= 8:
            self.toFile(stream)
    
    def flushBuffer(self, stream):
        # Add trailing zeros to complete a byte and write it
        if len(self.buffer) > 0:
            self.buffer += "0" * (8 - len(self.buffer))
            self.toFile(stream)
        assert(len(self.buffer) == 0)

    def encodeDims(self, band, stream):
        # Encode original image dimensions as padded 16 bit values (2 bytes each)
        rows = np.shape(band)[0]
        binRow = f'{rows:016b}'
        self.toBuffer(binRow, stream)

        cols = np.shape(band)[1]
        binCol = f'{cols:016b}'
        self.toBuffer(binCol, stream)

    def encodeTree(self, stream):
        # Compress tree and write to file
        root = self.heap[0]
        self.encodeTreeHelper(root, stream)
    
    def encodeTreeHelper(self, node, stream):
        if (node.left == None and node.right == None):
            # Leaf node, write 1, followed by value 
            stream.write((struct.pack('b', 1)))
            data = struct.pack('f', node.val)
            stream.write(data)
        else:
            # Parent node, write 0, continue down
            stream.write((struct.pack('b', 0)))
            self.encodeTreeHelper(node.left, stream)
            self.encodeTreeHelper(node.right, stream)

    def encodeImage(self, band, stream):
        # Write encoded pixel values to binary file
        for row in range(np.shape(band)[0]):
            for col in range(np.shape(band)[1]):
                self.toBuffer(self.codeBook[(band[row][col])], stream)

    def compress(self):
        fileNameW = 'compressedFrame.bin'
        outFile = open(fileNameW, 'wb')
        
        """
        Format of compressed BIN file:
            first 4 bytes = original image dimensions
            encoded huffman tree
            encoded image pixel values
        """

        freq = self.frequency(self.bandHH)
        self.createHeap(freq)
        self.buildTree()
        self.encodeDims(self.bandHH, outFile)
        self.encodeTree(outFile)
        self.buildCodebook()
        self.encodeImage(self.bandHH, outFile)
        self.flushBuffer(outFile)
        self.heap = []
        self.codeBook = {}

        freq = self.frequency(self.bandHL)
        self.createHeap(freq)
        self.buildTree()
        self.encodeDims(self.bandHL, outFile)
        self.encodeTree(outFile)
        self.buildCodebook()
        self.encodeImage(self.bandHL, outFile)
        self.flushBuffer(outFile)
        self.heap = []
        self.codeBook = {}

        freq = self.frequency(self.bandLH)
        self.createHeap(freq)
        self.buildTree()
        self.encodeDims(self.bandLH, outFile)
        self.encodeTree(outFile)
        self.buildCodebook()
        self.encodeImage(self.bandLH, outFile)
        self.flushBuffer(outFile)

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
        originalImage = np.zeros((originalRows, originalCols), dtype = np.float64)
        
        # Read in compressed values as binary string
        while byte := inFile.read(1):
            self.binaryStr += bin(int(byte.hex(), 16))[2:].zfill(8)

        # Use binaryStr to recreate original image
        decompressedImage = self.decompressPixels(originalImage, originalRows, originalCols, huffmanTree, inFile)

        # Check if reconstructed image is identical
        np.testing.assert_array_equal(decompressedImage, self.image)

    def decompressTree(self, stream):
        byte = stream.read(1)
        if struct.unpack('b',byte)[0] == 1:
            # Leaf node, return value
            packedValue = stream.read(4)
            value = struct.unpack('f',packedValue)[0]
            return value
        else:
            # Parent node, keep going
            left = self.decompressTree(stream)
            right = self.decompressTree(stream)
            return (left, right)

    def decompressPixels(self, image, rows, cols, tree, stream):
        # Restore each pixel value
        for x in range(rows):
            for y in range(cols):
                image[x][y] = self.decompressValue(tree, stream)
        return image

    def decompressValue(self, tree, stream):
        if (len(self.binaryStr)) == 0:
            return
        # Traverse the decompressed tree bit by bit until a value is hit
        bit = int(self.binaryStr[0])
        self.binaryStr = self.binaryStr[1:]
        node = tree[bit]
        if type(node) == tuple:
            return self.decompressValue(node, stream)
        else:
            return node



""" Testing """

originalHH = np.load('outfileHH.npy')
originalHL = np.load('outfileHL.npy')
originalLH = np.load('outfileLH.npy')

start = time.time()
huffmanCoding(originalHH, originalHL, originalLH).compress()
end = time.time()
firstC = end-start

"""
start = time.time()
huffmanCoding(originalHH, originalHL, originalLH).decompress()
end = time.time()
firstD = end-start
"""
print("-------------------------------------------------------------------")
#print("\nTime to compress|decompress HH: ", firstC, " | ", firstD)
print("-------------------------------------------------------------------")

