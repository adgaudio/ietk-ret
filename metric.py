import cv2
import numpy as np
from skimage import color
import matplotlib.pyplot as plt #importing matplotlib
from scipy import stats

def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

# Functions to comare histograms
def compare_hist(x1,x2,num):
    if(num==0):
    	# Kolmogorov–Smirnov test
        D,p = stats.ks_2samp(x1, x2)
    elif(num==1):
    	# Basically, it calculates the overlap between the two histograms and
    	# then normalizes it by second histogram (you can use first).
    	# You can use this function to calculate the similarity between the histograms.
        hist_1, _ = np.histogram(x1, bins=100, range=[0, 256])
        hist_2, _ = np.histogram(x2, bins=100, range=[0, 256])
        D = return_intersection(hist_1,hist_2)

    return D

# Give any imoriginal used after removing noise,
# imbinary images are labelled images from director "dir2" defined later
# Will return 2 arrays, i.e an array of pixel values which are good part of eye
# And onther array of pixels which represents bad part of eye
def get_good_bad_pixels(imoriginal,imbinary):

	bad_pixel = imoriginal[imbinary != 0]
    good_pixel = imbinary[imbinary == 0]
    good_pixel = good_pixel[good_pixel > 15]

    return bad_pixel, good_pixel

dir = '/home/ankit/Desktop/MLSP/project/indian-diabetic-retinopathy-dataset/indian-diabetic-retinopathy-dataset/A. Segmentation/1. Original Images/a. Training Set/'
dir2 = '/home/ankit/Desktop/MLSP/project/indian-diabetic-retinopathy-dataset/indian-diabetic-retinopathy-dataset/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages/'

file = 'IDRiD_02'
imoriginal = cv2.imread(dir+file+'.jpg')
imbinary = cv2.imread(dir2+file+'_HE.tif')[:, :, 2]

for ch in [0, 1, 2]:
    bad_pixel, good_pixel = get_good_bad_pixels(imoriginal[:, :, ch],imbinary)

    x = plt.hist(bad_pixel,bins=256)
    plt.xlim(0, 80) ; plt.title('bad pixel, channel: %s' % ch)
    plt.show()
    x = plt.hist(good_pixel,bins=256)
    plt.xlim(0, 80) ; plt.title('good pixel, channel: %s' % ch)
    plt.show()

    # Function to compare bad pixels with good pixels using metic defined
    metric = compare_hist(bad_pixel, good_pixel, 1)

    print(metric,"ch ",ch)
