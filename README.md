# Image-compression-and-repair

Down-sampled images could result from purposely compressing the images (to save computation resources) or accidental corruption of the images (failure of CCD component, for example).
It is almost unavoidable that we would need to cope with down-sampled images in research or in life. On the right is one example of sampled images. Therefore, it is imperative that we develop a systematic way to perform image recovery.

## How do we recover a sampled image?
We applied Discrete Cosine Transforms (DCT)1 to convert image recovery problem into regression problem. We then applied LASSO regularization to find one solution for this underdetermined system.
DCT expresses a function or a signal in terms of the sum of sinusoids with different frequency and amplitude. Just like the 1-D DCT Transform, in a 2-D context, we can generate 2-D DCT basis for the image size of choice based on the combination of each spatial frequency pair.
section 01
1. N. Ahmed, T. Natarajan and K. R. Rao, "Discrete Cosine Transform," in IEEE Transactions on Computers, vol. C-23, no. 1, pp. 90-93, Jan. 1974, doi: 10.1109/T-C.1974.223784.c
4
Rasterization (or flattening) of the 2-D DCT basis will result in a transformation matrix. Each column represents a basis, and our goal is to find the weight coefficients for each basis such that we can obtain the pixel value at any location of the image for reconstruction.
However, since the image is sampled, our system is underdetermined. So, we would need another constraint to find exactly one solution. We implemented LASSO regularization to constrain the parameters with sparsity, and the recovery of image is achieved by recombining all the small blocks and applying median filter for the continuous-looking image quality.
