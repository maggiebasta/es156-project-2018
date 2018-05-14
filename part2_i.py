import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import cv2

N = 35

def compute_dct(x):
	# Set empty array for dct coefficients
	dct = np.zeros((N, N), dtype=np.float64)

	# Iterate through and compute values for all coefficients
	for k in range(N):
		for l in range(N):

			# Iterate through and compute all values needed for sum of (k,l)th coefficient
			for m in range(N):

				# Set scalar factor values of coefficients
				a1 = 1.0/np.sqrt(2.0) if k == 0 else 1
				a2 = 1.0/np.sqrt(2.0) if l == 0 else 1

				# Compute value for (k,l)th coefficient
				for n in range(N):
					cm = np.cos((k*np.pi*(2*m+1)/(2*N)))
					cn = np.cos((l*np.pi*(2*n+1)/(2*N)))
					dct[k,l] += x[m,n]*a1*a2*cm*cn

	# normalize the transform
	dct = 2*dct/np.sqrt(N**2)
	return dct

def test():
	# Select image and compute dct 
	img = cv2.imread("sample_photos/dct_sample1.png", 0)
	dct = compute_dct(img)
	dct2 = fftpack.dct(fftpack.dct(img.T, type=2, norm='ortho').T, type=2, norm='ortho')


	# Plot the image next to the transform and the scipy transform
	plt.subplot(131),plt.imshow(img, cmap = 'gray')
	plt.xlabel('a', fontsize=8), plt.xticks([]), plt.yticks([])
	plt.subplot(132),plt.imshow(dct, cmap = 'gray')
	plt.xlabel('b', fontsize=8), plt.xticks([]), plt.yticks([])
	plt.subplot(133),plt.imshow(dct2, cmap = 'gray')
	plt.xlabel('c', fontsize=8), plt.xticks([]), plt.yticks([])
	plt.savefig('dct.png')
	plt.show()

def main():
	test()

if __name__ == '__main__':
	main()