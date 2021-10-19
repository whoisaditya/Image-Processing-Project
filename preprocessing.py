import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian

def add_gaussian_noise(img, sigma):
    gauss = np.random.normal(0, sigma, np.shape(img))

    noise_input = img + gauss
    noise_input[noise_input < 0] = 0
    noise_input[noise_input > 255] = 255

    return noise_input

def gaussian_kernel(kernel_size = 3):
    w = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    w = np.dot(w, w.transpose())
    w /= np.sum(w)
    return w

def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)

    demo = np.copy(img)
    demo = fft2(demo)

    kernel = fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)

    demo = demo * kernel
    demo = np.abs(ifft2(demo))
    #print (type(demo))
    return demo


