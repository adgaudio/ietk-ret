import scipy.ndimage as ndi
import evaluation as EV
import dehaze
import util
from idrid import IDRiD
from matplotlib import pyplot as plt
import metric
import multiprocessing
from itertools import repeat
import numpy as np
import scipy.optimize

from eval_separability import ks_scores_from_hist

NUM_IMG_SAMPLE = 30  # number of training sample loaded from IDRiD dataset, should be <= 54
NUM_ITER = 15  # number of training iteration for cmaes
NUM_WEIGHT_SAMPLE = 20  # number of samples generated at each cma iteration


class CMAES:
    def __init__(self, env, n, p, sigma, num_iter):
        """
        env: the environment object, should have a evaluate method to evaluate current weights
        n: number of weight samples generated at each iteration
        p: proportion of members used to update the mean and covariance
        sigma: initial std
        """
        self.env = env
        self.n = n
        self.p = p
        self.pn = int(self.p * self.n)
        self.sigma = sigma
        self.num_iter = num_iter
        self.d = env.num_weights   # at here the dimension is 9

        self.mu = np.zeros(self.d)
        self.S = sigma ** 2 * np.eye(self.d)

    def populate(self):
        """
        Populate a generation using the current estimates of mu and S

        Return:
        np array with size = self.d
        """
        return np.random.multivariate_normal(self.mu, self.S)

    def train(self):
        """
        Train CMA-ES self.num_iter iterations.
        For each iteration, generate self.n weights according to self.mu and self.S,
        then evaluate the weights to get the score,
        then update self.mu and self.S according to the top self.p proportion of members
        """

        for t in range(self.num_iter):
            weights_list = []
            score_list = []
            for i in range(self.n):
                weights = self.populate()
                score = self.env.evaluate(weights)

                weights_list.append(weights)
                score_list.append(score)

            # sort by score
            weights_list = np.array(weights_list)
            score_list = np.array(score_list)
            list_index = np.argsort(score_list, axis=0)[::-1]
            score_list = score_list[list_index]
            weights_list = weights_list[list_index]

            # update mu and S according to top self.pn score
            weights_update = weights_list[:self.pn]
            self.mu = np.mean(weights_update, axis=0)
            self.S = np.cov(weights_update.T)

            # log
            print("At iteration {}, average score: {}".format(t + 1, np.mean(score_list)))

    def get_result(self):
        """
        Get the result weights
        """
        return self.mu


"""
Represent the environment of IDRiD dataset
"""
class Environment:
    def __init__(self, num_workers, num_samples):
        self.images = []
        self.labels = []
        self.focus_regions = []
        self.num_workers = num_workers
        self.num_weights = 9  # number of weights of current environment

        dset = IDRiD('./data/IDRiD_segmentation')
        dset_iter = dset.iter_imgs(shuffle=True)

        for i in range(num_samples):
            _, img, labels = next(dset_iter)
            fg = util.get_foreground(img)[:,:,0]

            self.images.append(img)
            self.labels.append(labels)
            self.focus_regions.append(fg)

    def evaluate(self, weights):
        """
        Evaluate the score with given transform_matrix input

        Input:
        weights: size = self.num_weights, can be converted to transform_matrix
        """
        transform_matrix = weights.reshape(3, 3)
        # get evaluation images by performing transformation on each image
        evl_imgs = [get_transformed_img(x, y) for (x,y) in zip(self.images, repeat(transform_matrix))]

        # evaluate the result score of each image
        if self.num_workers == 1:
            result = [get_score_per_img(a,b,c) for a,b,c in zip(evl_imgs, self.labels, self.focus_regions)]
        else:
            with multiprocessing.Pool(processes=self.num_workers) as p:
                result = p.starmap(get_score_per_img, zip(evl_imgs, self.labels, self.focus_regions))

        return np.mean(result)


def get_transformed_img(img, transform_matrix):
    height, width, num_channels = img.shape
    img = img.reshape(-1, num_channels)

    img_transformed = np.dot(img, transform_matrix)

    return img_transformed.reshape(height, width, num_channels)


def get_score_per_img(img, labels, focus_region):
    """
    Return the average score of given image with its labels of all disease
    """
    #  a,b,c = (evl_imgs[0], self.labels[0], self.focus_regions[0][:, :, 0])
    ks_score = 0
    for mask in labels.values():
        a,b,c = img, mask, focus_region
        h = healthy_pixels = a[~b&c]
        d = diseased_pixels = a[b&c]
        H = np.histogramdd(h, bins=256, range=[(0,a.max())]*3)[0]
        D = np.histogramdd(d, bins=256, range=[(0,a.max())]*3)[0]
        _ks_score = np.abs(
            (H/H.sum()).ravel().cumsum() - (D/D.sum()).ravel().cumsum()).max()
        if not np.isnan(_ks_score):
            ks_score += _ks_score
        else:
            print('nan ks score due to transformed color channel being empty')
    return ks_score / len(labels)

def optimze_func(weights, env):
    return 1 - env.evaluate(weights)

if __name__ == "__main__":
    import sys
    try:
        num_cores = int(sys.argv[1])
    except:
        num_cores = multiprocessing.cpu_count()

    env = Environment(num_cores, NUM_IMG_SAMPLE)
    usingCMA = False

    if usingCMA:
        # use cmaes
        cma = CMAES(env, NUM_WEIGHT_SAMPLE, 0.5, 10, NUM_ITER)
        cma.train()
        transform_matrix = cma.get_result().reshape(3, 3)
    else:
        # using scipy.optimize
        result = scipy.optimize.minimize(optimze_func, np.random.rand(9), env, tol=0.1)
        print(result)
        transform_matrix = result.x.reshape(3,3)

    np.save('./transform_matrix.npy', transform_matrix)
