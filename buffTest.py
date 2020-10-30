import numpy as np
from doubleBuff import doubleBuffer

""" Testing by filling buffer with 2x2 numpy arrays """

# Set buffer size
size = 17
c = doubleBuffer(size)

index = 0
for x in range(size//2):
    # inserting random numpy arrays but the first element is numbered, in order, starting from 0
    # to be able to tell what array got added when
    testA = np.random.rand(2,2)
    testA[0,0] = index
    c.insert(testA)
    index += 1
# Testing when buffer is not full
print("Buffer contents: \n", c)
print("\nOrdered result: \n", c.orderedBuffer())
print("---------------------------------------")
print("---------------------------------------")

for x in range(size):
    testA = np.random.rand(2,2)
    testA[0,0] = index
    c.insert(testA)
    index += 1
# Testing when buffer is full
print("Buffer contents: \n", c)
print("\nOrdered result: \n", c.orderedBuffer())
print("---------------------------------------")
print("---------------------------------------")

for x in range(size//5):
    testA = np.random.rand(2,2)
    testA[0,0] = index
    c.insert(testA)
    index += 1
print("Buffer contents: \n", c)
print("\nOrdered result: \n", c.orderedBuffer())
