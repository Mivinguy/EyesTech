import numpy as np

class doubleBuffer(object):
    def __init__(self, size):
        # Initialization of double buffer
        self.writeIndex = 0
        self.readIndex = 0
        self.size = size
        self._data = []

    def insert(self, value):
        # Add frame to double buffer
        if len(self._data) == self.size:
            self._data[self.writeIndex] = value
            # Must increment readIndex if writeIndex overwrote data that readIndex is currently on
            if self.writeIndex == self.readIndex:
                self.readIndex = (self.readIndex + 1) % self.size
        else:
            self._data.append(value)
        self.writeIndex = (self.writeIndex + 1) % self.size

    def read(self):
        # Return frame at the current readIndex, helper function of orderedBuffer
        readResult = self._data[self.readIndex]
        self.readIndex = (self.readIndex + 1) % self.size
        return readResult

    def orderedBuffer(self):
        # Returns the entire double buffer sorted from oldest to newest frame 
        if len(self._data) == self.size:
            # Set readIndex onto oldest frame if it isnt already 
            # writeIndex is always on oldest frame after filling buffer
            #  (writeIndex - 1 % size) is always newest frame
            if self.readIndex != self.writeIndex:
                self.readIndex = self.writeIndex
            # Build result
            orderedFrames = []
            for x in range(self.size):
                orderedFrames.append(self.read())
            return orderedFrames
        else:
            return self._data

    def __getitem__(self, key):
        # Get frame
        return(self._data[key])

    def __repr__(self):
        # Return string representation
        return self._data.__repr__()
