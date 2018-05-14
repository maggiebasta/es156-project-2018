import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, dctn
import cv2
from part2_ii import partion_img, partitioned_dct, partitioned_dft

# Compute the quantized version of the specified transform (DCT or DFT)
def quantized_transform(img, k1, k2, T):

	# Takes dct coefficients as input and returns their quantized values
	def round(patches):

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
		I = patches.shape[0]
		J = patches.shape[1]
		K = patches.shape[2]
		L = patches.shape[3]

		# Initialize an empty matrix to store the quantized coefficients
		q_dct = np.zeros((I,J,K,L), dtype=np.float64)

		# Compute and store the quantized coefficients
		for i in range(I):
			for j in range(J):
				for k in range(K):
					for l in range(L):
						q_dct[i,j,k,l] = int(np.absolute(patches[i,j,k,l])/Qmtrx[k,l])
		return q_dct

	# Get DCT or DFT for the partioned image
	transforms = partitioned_dct(img, k1, k2) if T=='c' else partitioned_dft(img, k1, k2)

	# Return the quantized patches 
	return round(transforms)


def test():

	# get sample image
	im = cv2.imread("sample_photos/puppy3.png", 0)
	im = cv2.resize(im, (0,0), fx=0.5, fy=0.5) 

	p = partion_img(im)[0][0]

	qdct4  = quantized_transform(im, 2, 2, 'c')[0][0]
	qdct16 = quantized_transform(im, 4, 4, 'c')[0][0]
	qdct32 = quantized_transform(im, 8, 4, 'c')[0][0]

	plt.subplot(232),plt.imshow(p,cmap='gray'), plt.xlabel('a',fontsize=8), plt.xticks([]), plt.yticks([])
	plt.subplot(234),plt.imshow(qdct4, cmap='gray'), plt.xlabel('b',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.subplot(235),plt.imshow(qdct16,cmap='gray'), plt.xlabel('c',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.subplot(236),plt.imshow(qdct32,cmap='gray'), plt.xlabel('d',fontsize=8), plt.xticks([]),plt.yticks([])
	plt.savefig('quantized_corner' + str(0) + ',' + str(0) + '.png')
	plt.show()

def main():
	test()

if __name__ == '__main__':
	main()