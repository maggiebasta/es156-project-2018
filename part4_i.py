import numpy as np
import matplotlib.pyplot as plt
import cv2

N = 256

# Computes mean squared reconstruction error
def compute_MSE(img, img_new):
	error = 0
	for i in range(N):
		for j in range(N):
			error += (img[i][j] - img_new[i][j])**2
	return error/(N**2)

# Plots Reconstruction Errors and PSNR against iterations for the RNN, quantized DCT and quantized DFT
def plot(im):

	# Store K^2 values for x-axis
	K = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11,12,13, 14,15])

	r0 = cv2.imread("RNN_decoded/image_00.png", 0)	
	r1 = cv2.imread("RNN_decoded/image_01.png", 0)	
	r2 = cv2.imread("RNN_decoded/image_02.png", 0)	
	r3 = cv2.imread("RNN_decoded/image_03.png", 0)	
	r4 = cv2.imread("RNN_decoded/image_04.png", 0)	
	r5 = cv2.imread("RNN_decoded/image_05.png", 0)	
	r6 = cv2.imread("RNN_decoded/image_06.png", 0)	
	r7 = cv2.imread("RNN_decoded/image_07.png", 0)	
	r8 = cv2.imread("RNN_decoded/image_08.png", 0)	
	r9 = cv2.imread("RNN_decoded/image_09.png", 0)	
	r10 = cv2.imread("RNN_decoded/image_10.png", 0)	
	r11 = cv2.imread("RNN_decoded/image_11.png", 0)	
	r12 = cv2.imread("RNN_decoded/image_12.png", 0)	
	r13 = cv2.imread("RNN_decoded/image_13.png", 0)	
	r14 = cv2.imread("RNN_decoded/image_14.png", 0)	
	r15 = cv2.imread("RNN_decoded/image_15.png", 0)	

	rs = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15]

	# Compute and store the MSE for the RNN reconstructed imgs
	r_errors = np.asarray([compute_MSE(im, r) for r in rs])

	jpeg_max = cv2.imread("sample_photos/puppy3_small3.jpg", 0)
	j = compute_MSE(im, jpeg_max)
	jpeg_error = [j,j,j,j,j,j,j,j,j,j,j,j,j,j,j,j] 

	plt.plot(K, r_errors, 'b-s', label='RNN MSE')
	plt.plot(K, jpeg_error, 'y--', label='High JPEG Standard')
	plt.legend()
	plt.xlabel('Iterations')
	plt.ylabel('Reconstruction Error')
	plt.savefig('RNN_plot.png')
	plt.show()

	# Display sample reconstructed images and plot reconstruction error


def test():
	# Get sample image
	im = cv2.imread("sample_photos/puppy3_small.png", 0)

	plot(im)

def main():
	test()

if __name__ == '__main__':
	main()