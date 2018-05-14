from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

M = 512
N = 512

def noise_image(img, noise_scale):

	# Downsample
	def downsample(im):
		pix = im.load()
		dimg = Image.new('L', (M/2, N/2))
		dpix = dimg.load()
		for i in range(0, M, 2):
			for j in range(0, N,2):
				dpix[i/2, j/2] = pix[i,j]
		return dimg

	# Add noise
	def noise(im):
		# Access pixels
		pix = np.asarray(im, dtype=np.uint8)
		# Add noise
		noisy_img = pix + np.random.normal(0.0, noise_scale, pix.shape)
		# Covert noisy image pixel values to integers
		noisy_img = np.asarray(noisy_img, dtype=np.uint8)
		# Convert noisy image array back to image
		return Image.fromarray(noisy_img, 'L')

	return noise(downsample(img))


def filter_image(img, method):

	# Denoise the image with specified method (own created method, numpy or opencv)
	def denoise(im, method):
		# If specified method is for numpy denoise or opencv denoise
		if method != 'self':
			fimg = np.asarray(im.convert('L'), dtype=np.uint8)
			if method == 'cv': 
				p = cv2.fastNlMeansDenoising(fimg,10,7,21)
			else:
				p = ndimage.gaussian_filter(fimg, 1)

		# If specified method is for own created method
		else:
			p = np.asarray(im, dtype=np.uint8) + np.zeros((M/2,N/2), dtype=np.uint8)
			for i in range(1, M/2-1):
				for j in range(1, N/2-1):
					neighbors = np.asarray([p[i-1, j], p[i+1, j], p[i, j-1],p[i, j+1], p[i+1, j+1], p[i-1, j+1], p[i+1, j-1], p[i+1, j-1]])
					mindist = min(abs(p[i,j] - n) for n in neighbors)
					std = np.std(neighbors)
					if mindist > std:
						p[i, j] = np.mean(neighbors)

		return Image.fromarray(p, 'L')

	def resize(im):
		pix = im.load()
		fimg = Image.new('L', (M, N))
		fpix = fimg.load()
		for i in range(0, M):
			for j in range(0, N):
				fpix[i, j] = pix[i/2,j/2]
		return fimg

	return resize(denoise(img, method))


def error(original, filtered):
	orig = original.load()
	filt = filtered.load()
	error = 0
	for i in range(M):
		for j in range(N):
			error += (orig[i,j] - filt[i,j])**2
	return error/(M*N)

def test():
	test_imgs = []
	for i in range(1,5):

		# Get set of test images
		im1 = Image.open("sample_photos/puppy" + str(i) + ".png").convert('L')
		im2 = Image.open("sample_photos/kitten" + str(i) + ".png").convert('L')
		test_imgs.append(im1)
		test_imgs.append(im2)

	# Select an image for testing
	img = test_imgs[0]

	# Create, show and save samples of noisy and filtered image
	noisy = noise_image(img, 4.0)
	filtered = filter_image(noisy, 'self')
	img.show()
	noisy.show()
	filtered.show()
	noisy.save('noisy.png')
	filtered.save('filtered.png')

	# Plot reconstruction error with varying noise for different denoise methods
	noise_amounts = np.linspace(0,16,8, dtype=np.float64)
	cv_errors = []
	np_errors = []
	self_errors = []
	for i in noise_amounts:
		noisy = noise_image(img, i)
		cv_filter = filter_image(noisy, 'cv')
		np_filter = filter_image(noisy, 'np')
		self_filter = filter_image(noisy, 'self')
		cv_errors.append(error(img, cv_filter))
		np_errors.append(error(img, np_filter))
		self_errors.append(error(img, self_filter))

	# plt.plot(noise_amounts, self_errors, noise_amounts, cv_errors, noise_amounts, np_errors)
	plt.plot(noise_amounts, self_errors, 'bs--', label='denoiser')
	plt.plot(noise_amounts, cv_errors, 'ro--', label='opencv denoiser')
	plt.plot(noise_amounts, np_errors, 'g^--', label='numpy denoiser')
	plt.legend()
	plt.xlabel('Gaussian Noise Added')
	plt.ylabel('Reconstruction Error')
	plt.savefig('noise_reconstruction_errors.png')
	plt.show()

def main():
	test()

if __name__ == '__main__':
	main()

