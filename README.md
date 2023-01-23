# Image-compression-and-repair

Down-sampled images could result from purposely compressing the images (to save computation resources) or accidental corruption of the images (failure of CCD component, for example).
It is almost unavoidable that we would need to cope with down-sampled images in research or in life. On the right is one example of sampled images. Therefore, it is imperative that we develop a systematic way to perform image recovery.
![Screen Shot 2023-01-23 at 12 29 00 AM](https://user-images.githubusercontent.com/90811429/213971533-d65a0219-adc9-4c84-934d-0231d80559f7.png)

## How do we recover a sampled image?
We applied Discrete Cosine Transforms (DCT)1 to convert image recovery problem into regression problem. We then applied LASSO regularization to find one solution for this underdetermined system.
DCT expresses a function or a signal in terms of the sum of sinusoids with different frequency and amplitude. Just like the 1-D DCT Transform, in a 2-D context, we can generate 2-D DCT basis for the image size of choice based on the combination of each spatial frequency pair.

Rasterization (or flattening) of the 2-D DCT basis will result in a transformation matrix. Each column represents a basis, and our goal is to find the weight coefficients for each basis such that we can obtain the pixel value at any location of the image for reconstruction.
However, since the image is sampled, our system is underdetermined. So, we would need another constraint to find exactly one solution. We implemented LASSO regularization to constrain the parameters with sparsity, and the recovery of image is achieved by recombining all the small blocks and applying median filter for the continuous-looking image quality.
## How do we recover a sampled image?
Introducing a new constraint: LASSO regularization.Linear regression alone does not work in our case because the image is sampled. The unknowns in some of the target pixel values render our system to be underdetermined.As we can see in Fig. 2(b), we do not have enough observations but have too many unknowns, leaving us an underdetermined system that has infinitely many solutions. Here we will make use of the sparse nature of natural images: For small natural images, there tends to be small number of non-zero coefficients.The mathematical implication of sparsity is to impose a L1- norm regularization constraint to our regression problem, which is exactly what LASSO Regularization2 does. Since it constrains the weight coefficients to be mostly zeros, we again convert the regression problem into a convex optimization problem:
## How to select regularization constant?
As with normal cases, the constant selection is done by cross-validation.3 However, we will cross-validate on random subsets of the sampled pixels of the block.
In our fishing boat image example, we use 1/6 of the sampled pixels for the test set and use the remaining 5/6 of the sampled pixels as the training set. The pixels are drawn randomly, and the process repeats 20 times to obtain our optimized regularization constant.
For each of the 20 repetition, we measure the error on the test set with mean squared error. We then average the error over 20 rounds, keeping track of this mean error for every possible regularization constant in our search range. (10−6 𝑡𝑜 106 in this case) The one with the least mean error will be our constant of choice.
Once the regularization constant 𝜆 is found, the following equation will give us the weight coefficients vector that we are after. We then reconstruct the 8 x 8 block in the same way we rasterize it.
