from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import cv2

M = 512
N = 512

def compute_fourier(img, alpha):
	# Compute discrete fourier for the image
	cfs = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
	# Compute the magnitudes of the coefficients
	mags = 20*np.log(cv2.magnitude(cfs[:,:,0],cfs[:,:,1]))
	# Calculate the number of coefficients to replace with zero
	num_removed = int(M*N*(1.0-alpha))

	for i in range(num_removed):
		# Find the indices of the smallest coefficient
		cmin = np.unravel_index(np.argmin(mags), (M,N))
		# Set the smallest coefficient value to zero
		cfs[cmin] = 0
		# Change coefficient value in magnitude array for next iteration
		mags[cmin] = (float('inf'))

	# Return alpha% largest coefficients
	return cfs


def inverse_fourier(cfs):
	inverse = cv2.idft(cfs)
	# Get the magnitude component of the inverted transform
	mags = cv2.magnitude(inverse[:,:,0],inverse[:,:,1])/(M*N)

	# Return tuple of the inverse fourier and the magnitudes
	return inverse, mags


# Computes reconstruction error
def compute_error(img, img_new):
	error = 0
	for i in range(M):
		for j in range(N):
			error += (img[i][j] - img_new[i][j])**2
	return error/(M*N)

# Plots reconstruction error vs alpha
def plot(img):
	alphas = np.linspace(.01,.99,99, dtype=np.float64)
	errors = np.zeros(len(alphas))
	for a in range(len(alphas)):
		cfs = compute_fourier(img, alphas[a])
		inverse_mag = inverse_fourier(cfs)[1]
		errors[a] = compute_error(img,inverse_mag)

	plt.plot(alphas, errors, 'o')
	plt.xlabel('alpha')
	plt.ylabel('Reconstruction Error')
	plt.title('FT Compression/Distortion Trade-Off')
	plt.show()


def test():
	test_imgs = []
	# Get set of test images
	for i in range(1,5):
		im1 = cv2.imread("sample_photos/puppy" + str(i) + ".png", 0)
		im2 = cv2.imread("sample_photos/kitten" + str(i) + ".png", 0)
		test_imgs.append(im1)
		test_imgs.append(im2)



	# Select image, compute transform and compute inverse
	img = test_imgs[3]

	# Plot reconstruction error
	plot(img)

	# Get samples of reconstructed images
	coeffs1 = compute_fourier(img, .5)
	inverse_mag1 = inverse_fourier(coeffs1)[1]
	coeffs2 = compute_fourier(img, .01)
	inverse_mag2 = inverse_fourier(coeffs2)[1]

	# plot original image next to reconstructed images
	plt.subplot(311),plt.imshow(img, cmap = 'gray')
	plt.xlabel('a', fontsize=8), plt.xticks([]), plt.yticks([])
	plt.subplot(312),plt.imshow(inverse_mag1, cmap = 'gray')
	plt.xlabel('b', fontsize=8), plt.xticks([]), plt.yticks([])
	plt.subplot(313),plt.imshow(inverse_mag2, cmap = 'gray')
	plt.xlabel('c', fontsize=8), plt.xticks([]), plt.yticks([])
	plt.savefig('compression_set.png')
	plt.show()

def main():
	test()

if __name__ == '__main__':
	main()

