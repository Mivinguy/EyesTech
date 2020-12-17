###########

#

# Padding arrays for DWT V1.0

# 12/15/2020

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
    newArr = np.zeros((newRowNum,newColNum), dtype = np.int16)
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
