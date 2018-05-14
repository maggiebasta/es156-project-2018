import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, dctn
import cv2

# Global value for test image dimension
N = 256

# Global value for the number of patches
prange = N/8

def partion_img(img):
	# Set empty array for partioned image
	patches = np.zeros((prange,prange, 8, 8))
	# Split the image into patches of 8x8 
	for i in range(0, N, 8):
		for j in range(0, N, 8):
			patches[i/8,j/8] = img[i:i+8, j:j+8]
	return patches

def partitioned_dft(img, K1, K2):
	# Get set of 8x8 patches of the image
	partioned = partion_img(img)
	# Set empty array of sizes K^2 per patch
	dft_patches = np.zeros((prange,prange, K1, K2), dtype=np.complex64)
	for i in range(partioned.shape[0]):
		for j in range(partioned.shape[1]):
			dft_patches[i,j] = np.fft.fft2(partioned[i,j], s=[K1,K2])
	return dft_patches

def partitioned_dct(img, K1, K2):
	# Get set of 8x8 patches of the image
	partioned = partion_img(img)
	# Set empty array of sizes K^2 per patch
	dct_patches = np.zeros((prange,prange, K1, K2), dtype=np.float64)
	for i in range(partioned.shape[0]):
		for j in range(partioned.shape[1]):	
			dct_patches[i,j] = dctn(partioned[i,j], type=2, shape=[K1,K2], norm='ortho')
	return dct_patches


# Takes an image and a patch index and plots patch w/ dft and dct for K^2 = 4,16,64
def plot_patch_data(im, i, j):
	
	# Get the patch
	p = partion_img(im)[i][j]

	# Compute the transforms for the patch
	fmags4 = [np.absolute(f) for f in partitioned_dft(im, 2, 2)[i][j]]	
	fmags16 = [np.absolute(f) for f in partitioned_dft(im, 4, 4)[i][j]]
	fmags64 = [np.absolute(f) for f in partitioned_dft(im, 8, 4)[i][j]]
	dct4  = partitioned_dct(im, 2, 2)[i][j]
	dct16 = partitioned_dct(im, 4, 4)[i][j]
	dct32 = partitioned_dct(im, 8, 4)[i][j]

	# Plot the patch and the transforms
	plt.subplot(332),plt.imshow(p,cmap='gray'), plt.xlabel('a',fontsize=8), plt.xticks([]), plt.yticks([])
	plt.subplot(334),plt.imshow(fmags4, cmap='gray'), plt.xlabel('b',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.subplot(335),plt.imshow(fmags16,cmap='gray'), plt.xlabel('c',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.subplot(336),plt.imshow(fmags64,cmap='gray'), plt.xlabel('d',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.subplot(337),plt.imshow(dct4, cmap='gray'), plt.xlabel('e',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.subplot(338),plt.imshow(dct16,cmap='gray'), plt.xlabel('f',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.subplot(339),plt.imshow(dct32,cmap='gray'), plt.xlabel('g',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.show()
	plt.savefig('partioned_corner' + str(i) + ',' + str(j) + '.png')


def test():

	# Get sample image
	im1 = cv2.imread("sample_photos/puppy3.png", 0)
	im1 = cv2.resize(im1, (0,0), fx=0.5, fy=0.5) 

	# Plot samples for partitioned tranforms 
	plot_patch_data(im1, 0, 0)
	plot_patch_data(im1, 31, 31)

def main():
	test()

if __name__ == '__main__':
	main()