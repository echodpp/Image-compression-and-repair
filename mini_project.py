import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from scipy import ndimage
from scipy.fft import fft, dct, idct
from math import cos, sqrt, pi
from PIL import Image
from sklearn.metrics import mean_squared_error

from sklearn import linear_model
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    ShuffleSplit,
    GridSearchCV,
)


def imgRead(fileName, cnvt=None):
    """
    load the input image into a matrix
    :param fileName: name of the input file
    :param cnvt: convert image into certain type, default=None
    :return: a matrix of the input image
    """
    imgIn = Image.open(fileName).convert(cnvt)
    imgToArr = np.array(imgIn)
    return imgToArr


def imgShow(imgOutArr, cmap="viridis", vmin=0, vmax=255):
    """
    show the image saved in a matrix
    :param imgOut: a matrix containing the image to show
    :param cmap: how the image is shown
    :return: None
    """
    #     imgOut = np.uint8(imgOutArr)  # What does this line do and is it necessary?
    plt.imshow(imgOutArr, cmap=cmap, vmin=vmin, vmax=vmax)


# if __name__ == '__main__':
#     a = imgRead('lena.bmp', 'L')
#     print(np.shape(a))
#     imgShow(a, 'gray')
#     print(a)


def imgSample(imgIn, numSample):
    """
    Sample the input image
    :param imgIn: input image
    :param numSample: how many samples in this image
    :return: sampled image
    """
    assert (
        numSample >= 0 and numSample <= imgIn.shape[0] * imgIn.shape[1]
    ), "Sampling exceeds upper limit"
    ans = np.empty(imgIn.shape)
    ans.fill(np.nan)
    s = numSample
    while s != 0:
        Ind1 = np.random.randint(0, imgIn.shape[0])
        Ind2 = np.random.randint(0, imgIn.shape[1])
        if np.isnan(ans)[Ind1, Ind2]:
            ans[Ind1, Ind2] = imgIn[Ind1, Ind2]
            s -= 1
        else:
            continue
    return ans


def imgSlice(imgIn, blkSize):
    h, w = imgIn.shape
    assert (
        h % blkSize == 0 and w % blkSize == 0
    ), "Image is not divisible by {}x{} block".format(blkSize, blkSize)
    return (
        imgIn.reshape(h // blkSize, blkSize, -1, blkSize)
        .swapaxes(1, 2)
        .reshape(-1, blkSize, blkSize)
    )


def recombine(imgBlks, h, w):
    n, nrows, ncols = imgBlks.shape
    return imgBlks.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)


# Reference: https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755


def img2DCT(imgIn):
    return dct(dct(imgIn, axis=0, norm="ortho"), axis=1, norm="ortho")


def img2IDCT(coef):
    return idct(idct(coef, axis=0, norm="ortho"), axis=1, norm="ortho")


def imgDCT(imgIn):
    return dct(imgIn, norm="ortho")


def imgIDCT(coef):
    return idct(coef, norm="ortho")


# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html


def customDCT(x, y, u, v, n):
    # Normalisation
    def alpha(a):
        if a == 0:
            return sqrt(1.0 / n)
        else:
            return sqrt(2.0 / n)

    return (
        alpha(u)
        * alpha(v)
        * cos(((2 * x + 1) * (u * pi)) / (2 * n))
        * cos(((2 * y + 1) * (v * pi)) / (2 * n))
    )


def getBasisImage(u, v, n):
    # for a given (u,v), make a DCT basis image
    basisImg = np.zeros((n, n))
    for y in range(0, n):
        for x in range(0, n):
            basisImg[y, x] = customDCT(x, y, u, v, n)
    return basisImg


def getDCTBasis(n):
    imageSet = []
    for u in range(0, n):
        for v in range(0, n):
            basisImg = getBasisImage(u, v, n)
            imageSet.append(basisImg)
    return np.array(imageSet)


def transformDCTMatrix(imageSet):
    num, h, w = imageSet.shape
    return imageSet.reshape(-1, h * w).T


# Reference: https://github.com/chalmersgit/Discrete-Cosine-Transform/blob/master/dct.py


def getDCTCoef(imgIn, blk):
    assert (
        imgIn.shape[0] == blk and imgIn.shape[1] == blk
    ), "Mismatch in image size and block size"
    T = transformDCTMatrix(getDCTBasis(blk))
    arr = imgIn.reshape(-1)
    coef = np.linalg.inv(T) @ arr
    return coef


def imgRecover(imgIn, blkSize, numSample):
    """
    Recover the input image from a small size samples
    :param imgIn: input image
    :param blkSize: block size
    :param numSample: how many samples in each block
    :return: recovered image
    """
    # Preprocess Image
    arrBlk = imgSlice(imgIn, blkSize)
    sampledBlk = np.array([imgSample(b, numSample) for b in arrBlk])
    # Construct DCT basis
    basis2D = getDCTBasis(blkSize)  # list of 2D images
    basis1D = transformDCTMatrix(basis2D)
    # Lasso Regression with random subset cv
    cv = ShuffleSplit(n_splits=20, test_size=numSample // 6, random_state=0)
    recoverArr = []
    for blk in sampledBlk:
        b0 = blk.reshape(-1)
        mask = np.invert(np.isnan(b0))
        b_s = b0[mask]
        basis_s = basis1D[mask]
        clf = linear_model.LassoCV(
            alphas=np.logspace(-6, +6, 300),
            fit_intercept=False,
            cv=cv,
            max_iter=100000,
            n_jobs=-1,
        )
        clf.fit(basis_s, b_s)
        result = (basis1D @ clf.coef_).reshape(blkSize, -1)
        recoverArr.append(result)
    combineImg = recombine(np.array(recoverArr), imgIn.shape[0], imgIn.shape[1])
    return combineImg


def applyMedFilter(imgIn, sz=3):
    return ndimage.median_filter(imgIn, size=sz)


def imgMSE(imageA, imageB):
    # Check dimensions to be the same
    assert len(imageA) == len(imageB), "Mismatch in image's height"
    assert len(imageA[0]) == len(imageB[0]), "Mismatch in image's width"

    return mean_squared_error(imageA, imageB)


def compareImg(original, restoreImg, filtImg, show_image=True):
    print(
        "MSE of original image and recovered image w/o medfilt: {}".format(
            mean_squared_error(original, restoreImg)
        )
    )
    print(
        "MSE of original image and recovered image w/ medfilt: {}".format(
            mean_squared_error(original, filtImg)
        )
    )
    if show_image:
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(original, cmap="gray")
        plt.title("Original Image")
        plt.subplot(1, 3, 2)
        plt.imshow(restoreImg, cmap="gray", vmin=0, vmax=255)
        plt.title("Restored Image")
        plt.subplot(1, 3, 3)
        plt.imshow(filtImg, cmap="gray", vmin=0, vmax=255)
        plt.title("Restored Image + MedFilt")
