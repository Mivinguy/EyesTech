###########

#

# PADDING FUNCTION

#

###########

import numpy as np


def padArray(arr, dimensions):
    row, col = dimensions
    padAmount = 2     # Default 

    # Generate padded array dimensions
    newColNum = col + padAmount*2 + (col+1)%2  
    newRowNum = row + padAmount*2 + (row+1)%2   
    # if col is even, (col+1)%2 = 1 (adding extra col)
    # for odd, (col+1)%2 = 0 (just pads by 2)
    # used this same idea when padding right and below 

    # Allocate a new array of correct size and copy the original array into it
    newArr = np.zeros((newRowNum,newColNum), dtype = np.int8)
    newArr[padAmount:row+padAmount, padAmount:col+padAmount] = arr

    # Padding the left of the array symmetrically
    newArr[padAmount:padAmount+row, 0:padAmount] = np.fliplr(newArr[padAmount:padAmount+row, padAmount:padAmount+2])
    # Right || Deals with even and odd cases
    newArr[padAmount:padAmount+row, padAmount+col:newColNum] = np.fliplr(newArr[padAmount:padAmount+row, (col-(col+1)%2):col+padAmount])
    # Above
    newArr[0:padAmount, 0:newColNum] = np.flipud(newArr[padAmount:2*padAmount,])
    # Below || Deals with even and odd cases
    newArr[padAmount+row:newRowNum, 0:newColNum] = np.flipud(newArr[(row-(row+1)%2):row+padAmount,])

    return newArr


"""Testing code

# Adjust size of input array here
original = np.random.randint(255, size=(6, 5))
print("Original: \n", original, "\n")

symmetricOriginal = padArray(original, np.shape(original))
print("My generated array: \n", symmetricOriginal, "\n")

# Verify that the function is identical
# Only verifys when both dimensions are odd
if ((np.shape(original)[0]%2 != 0 and np.shape(original)[1])%2 != 0):
    numpyPad = np.pad(original, (2,), "symmetric")
    print("Numpy generated array: \n", numpyPad)
    print("\nDouble Odd Case (assert tested)\n")
    np.testing.assert_array_equal(symmetricOriginal, numpyPad)

# Only verifys when both dimensions are even
elif ((np.shape(original)[0]%2 == 0 and np.shape(original)[1]%2 == 0)):
    numpyPad = np.pad(original, (2,3), "symmetric")
    print("Numpy generated array: \n", numpyPad)
    print("\nDouble Even Case (assert tested)\n")
    np.testing.assert_array_equal(symmetricOriginal, numpyPad)

else:
    print("Single Even Case (no assert test)")
    numpyPad = np.pad(original, (2,), "symmetric")
    print("Numpy generated array for reference (missing a row or column): \n", numpyPad)
"""
