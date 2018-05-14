import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, dctn, idctn
import cv2
from part2_ii import partion_img, partitioned_dct, partitioned_dft
from part2_iii import quantized_transform

# Global value for test image dimension
N = 256

# Global value for the number of patches
prange = N/8


# Reconstructs the original image from recovered set of patches
def stitch(patches):
	recovered = np.zeros((N,N), dtype=np.float64)

	for i in range(0, prange):
		for j in range(0, prange):
			recovered[i*8:i*8+8,j*8:j*8+8] = patches[i][j]
	return recovered

# Computes iDCTs for patches
def inverse_partitioned(patches, T):

	# Set empty array for recovered set of patches image
	i_patches = np.zeros((prange,prange, 8, 8), dtype=np.float64)

	for i in range(prange):
		for j in range(prange):
			if T == 'c':
				i_patches[i,j] = idctn(patches[i,j], type=2, shape=[8,8], norm='ortho')
			else:
				ifft = np.fft.ifft2(patches[i,j], s=[8,8])
				i_patches[i,j] = [np.absolute(k) for k in ifft]

	return i_patches


# Unquantizes and computes iDCTs for patches
def inverse_quantized(qdct_patches, T):

	# Set empty array for recovered set of patches image
	i_patches = np.zeros((prange,prange, 8, 8), dtype=np.float64)

	# Takes rounded dct coefficients and returns their unquantized values
	def unround(dct_patches):

		# Initialize the quantization matrix
		Qmtrx = np.zeros((8,8))
		Qmtrx[0,:] = [16, 11, 10, 16, 24, 40, 51, 61]
		Qmtrx[1,:] = [12, 12, 14, 19, 26, 58, 60, 55]
		Qmtrx[2,:] = [14, 13, 16, 24, 40, 57, 69, 56]
		Qmtrx[3,:] = [14, 17, 22, 29, 51, 87, 80, 62]
		Qmtrx[4,:] = [18, 22, 37, 56, 68, 109, 103, 77]
		Qmtrx[5,:] = [24, 36, 55, 64, 81, 104, 113, 92]
		Qmtrx[6,:] = [49, 64, 78, 87, 103, 121, 120, 101]
		Qmtrx[7,:] = [72, 92, 95, 98, 112, 100, 103, 99]

		# Store dimensions of DCT coefficient matrix patches for iteration
		I = dct_patches.shape[0]
		J = dct_patches.shape[1]
		K = dct_patches.shape[2]
		L = dct_patches.shape[3]

		# Initialize an empty matrix to store the quantized coefficients
		iq_dct = np.zeros((I,J,K,L), )

		# Compute and store the quantized coefficients
		for i in range(I):
			for j in range(J):
				for k in range(K):
					for l in range(L):
						iq_dct[i,j,k,l] = int(dct_patches[i,j,k,l]*Qmtrx[k,l])
		return iq_dct

	return inverse_partitioned(unround(qdct_patches), T)

# Computes mean squared reconstruction error
def compute_MSE(img, img_new):
	error = 0
	for i in range(N):
		for j in range(N):
			error += (img[i][j] - img_new[i][j])**2
	return error/(N**2)

# Computes PSNR reconstruction error
def compute_PSNR(img, img_new):
	MSE = compute_MSE(img, img_new)
	PSNR = 10*np.log10(255**2/MSE)

	return PSNR


# Plots Reconstruction Errors and PSNR against K^2 for the DCT, quantized DCT and quantized DFT
def plot(im):

	# Store K^2 values for x-axis
	K = [4,16,32,64]	

	# Reconstruct using the partioned DCT for K^2 = 4, 16, 32, 64
	dc_new4  = stitch(inverse_partitioned(partitioned_dct(im, 2, 2), 'c'))
	dc_new16  = stitch(inverse_partitioned(partitioned_dct(im, 4, 4), 'c'))
	dc_new32  = stitch(inverse_partitioned(partitioned_dct(im, 8, 4), 'c'))
	dc_new64  = stitch(inverse_partitioned(partitioned_dct(im, 8, 8), 'c'))

	# Reconstruct using the quantized partioned DCT for K^2 = 4, 16, 32, 64
	dcq_new4  = stitch(inverse_quantized(quantized_transform(im, 2, 2, 'c'), 'c'))
	dcq_new16  = stitch(inverse_quantized(quantized_transform(im, 4, 4, 'c'), 'c'))
	dcq_new32  = stitch(inverse_quantized(quantized_transform(im, 8, 4, 'c'), 'c'))
	dcq_new64  = stitch(inverse_quantized(quantized_transform(im, 8, 8, 'c'), 'c'))

	# Reconstruct using the partioned DFT for K^2 = 4, 16, 32, 64
	df_new4  = stitch(inverse_partitioned(partitioned_dft(im, 2, 2), 'f'))
	df_new16  = stitch(inverse_partitioned(partitioned_dft(im, 4, 4), 'f'))
	df_new32  = stitch(inverse_partitioned(partitioned_dft(im, 8, 4), 'f'))
	df_new64  = stitch(inverse_partitioned(partitioned_dft(im, 8, 8), 'f'))

	# Reconstruct using the quantized partioned DFT for K^2 = 4, 16, 32, 64
	dfq_new4  = stitch(inverse_quantized(quantized_transform(im, 2, 2, 'f'), 'f'))
	dfq_new16  = stitch(inverse_quantized(quantized_transform(im, 4, 4, 'f'), 'f'))
	dfq_new32  = stitch(inverse_quantized(quantized_transform(im, 8, 4, 'f'), 'f'))
	dfq_new64  = stitch(inverse_quantized(quantized_transform(im, 8, 8, 'f'), 'f'))

	# Compute and store the MSE reconstruction errors for the DCT
	dc_errors = [compute_MSE(im, d) for d in [dc_new4, dc_new16, dc_new32, dc_new64]]
	dcq_errors = [compute_MSE(im, q) for q in [dcq_new4, dcq_new16, dcq_new32, dcq_new64]]

	# Compute and store the PSNR for the DFT
	dc_psnr = [compute_PSNR(im, d) for d in [dc_new4, dc_new16, dc_new32, dc_new64]]
	dcq_psnr = [compute_PSNR(im, q) for q in [dcq_new4, dcq_new16, dcq_new32, dcq_new64]]

	# Plot the Reconstruction Errors against K^2 for the DCT
	plt.plot(K, dc_errors, 'r-s', label='DCT')
	plt.plot(K, dcq_errors, 'b-s', label='Quantized DCT')
	plt.legend()
	plt.xlabel('K^2')
	plt.ylabel('Reconstruction Error')
	plt.title('DCT Reconstruction Error for Varying K^2')
	plt.savefig('DCT_error_plot.png')
	plt.show()

	# Plot the PSNR Errors against K^2 for the DCT
	plt.plot(K, dc_psnr, 'r-s', label='DCT')
	plt.plot(K, dcq_psnr, 'b-s', label='Quantized DCT')
	plt.legend()
	plt.xlabel('K^2')
	plt.ylabel('Reconstruction Error')
	plt.title('DCT PSNR for Varying K^2')
	plt.savefig('DCT_PSNR_plot.png')
	plt.show()

	# Compute and store the MSE reconstruction errors for the DFT
	df_errors = [compute_MSE(im, d) for d in [df_new4, df_new16, df_new32, df_new64]]
	dfq_errors = [compute_MSE(im, q) for q in [dfq_new4, dfq_new16, dfq_new32, dfq_new64]]

	# Compute and store the PSNR
	df_psnr = [compute_PSNR(im, d) for d in [df_new4, df_new16, df_new32, df_new64]]
	dfq_psnr = [compute_PSNR(im, q) for q in [dfq_new4, dfq_new16, dfq_new32, dfq_new64]]


	# Plot the MSE Reconstruction Errors against K^2 for the DFT and DCT
	plt.plot(K, df_errors, 'g--o', label='DFT')
	plt.plot(K, dfq_errors, 'k--o', label='Quantized DFT')
	plt.plot(K, dc_errors, 'r--s', label='DCT')
	plt.plot(K, dcq_errors, 'b--s', label='Quantized DCT')
	plt.legend()
	plt.xlabel('K^2')
	plt.ylabel('Reconstruction Error')
	plt.title('DFT and DCT Reconstruction Error for Varying K^2')
	plt.savefig('DFT_DCT_error_plot.png')
	plt.show()

	# Plot the PSNR Errors against K^2 for the DFT and DCT
	plt.plot(K, df_psnr, 'g--o', label='DFT')
	plt.plot(K, dfq_psnr, 'k--o', label='Quantized DFT')
	plt.plot(K, dc_psnr, 'r--s', label='DCT')
	plt.plot(K, dcq_psnr, 'b--s', label='Quantized DCT')
	plt.legend()
	plt.xlabel('K^2')
	plt.ylabel('Reconstruction Error')
	plt.title('DFT and DCT PSNR for Varying K^2')
	plt.savefig('DFT_DCT_PSNR_plot.png')
	plt.show()


	# compute jpeg standard errors for the image
	jpeg1 = cv2.imread("sample_photos/puppy3_small1.jpg", 0)
	j1 = compute_MSE(im, jpeg1)
	j1 = [j1, j1, j1, j1]
	jpeg2 = cv2.imread("sample_photos/puppy3_small2.jpg", 0)
	j2 = compute_MSE(im, jpeg2)
	j2 = [j2, j2, j2, j2]
	jpeg3 = cv2.imread("sample_photos/puppy3_small3.jpg", 0)
	j3 = compute_MSE(im, jpeg3)
	j3 = [j3, j3, j3, j3]
	jpeg4 = cv2.imread("sample_photos/puppy3_small4.jpg", 0)
	j4 = compute_MSE(im, jpeg4)
	j4 = [j4, j4, j4, j4]

	# Plot the MSE Reconstruction Errors against the Jpeg standard
	plt.plot(K, dc_errors, 'r-s', label='DCT')
	plt.plot(K, dcq_errors, 'b-s', label='Quantized DCT')
	plt.plot(K, j1, 'y--', label='JPEG Standards')
	plt.plot(K, j2,'y--', K, j3, 'y--', K, j4, 'y--')
	plt.legend()
	plt.xlabel('K^2')
	plt.ylabel('Reconstruction Error')
	plt.title('DFT and DCT Reconstruction Error for Varying K^2')
	plt.savefig('JPEG_standard_plot.png')
	plt.show()



# Display sample reconstructed images and plot reconstruction error
def test():
	# Get sample image
	im = cv2.imread("sample_photos/puppy3_small.png", 0)


	# Reconstruct using the partioned DCT for K^2 = 4, 16, 32, 64
	d_new4  = stitch(inverse_partitioned(partitioned_dct(im, 2, 2), 'c'))
	d_new16  = stitch(inverse_partitioned(partitioned_dct(im, 4, 4), 'c'))
	d_new32  = stitch(inverse_partitioned(partitioned_dct(im, 8, 4), 'c'))

	# Reconstruct using the quantized partioned DCT for K^2 = 4, 16, 32, 64
	q_new4  = stitch(inverse_quantized(quantized_transform(im, 2, 2, 'c'), 'c'))
	q_new16  = stitch(inverse_quantized(quantized_transform(im, 4, 4, 'c'), 'c'))
	q_new32  = stitch(inverse_quantized(quantized_transform(im, 8, 4, 'c'), 'c'))


	# Show the reconstructed images using the partioned DCT
	plt.subplot(131),plt.imshow(d_new4,cmap='gray'), plt.xlabel('a',fontsize=8), plt.xticks([]), plt.yticks([])
	plt.subplot(132),plt.imshow(d_new16, cmap='gray'), plt.xlabel('b',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.subplot(133),plt.imshow(d_new32,cmap='gray'), plt.xlabel('c',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.savefig('reconstructed_dct.png')
	plt.show()

	# Show the reconstructed images using the quantized partioned DCT
	plt.subplot(131),plt.imshow(q_new4,cmap='gray'), plt.xlabel('a',fontsize=8), plt.xticks([]), plt.yticks([])
	plt.subplot(132),plt.imshow(q_new16, cmap='gray'), plt.xlabel('b',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.subplot(133),plt.imshow(q_new32,cmap='gray'), plt.xlabel('c',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.savefig('reconstructed_qdct.png')
	plt.show()

	plot(im)


def main():
	test()

if __name__ == '__main__':
	main()