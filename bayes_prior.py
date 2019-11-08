"""
illumination with bayesian prior  - use the equation to increase brightness of
healthy pixels and decrease disease pixels.  this is possible since the depth
map can be thought of as probabilities.
"""
import numpy as np
from matplotlib import pyplot as plt
import pickle


from idrid import IDRiD
import util


def iter_imgs(labels):
    dset = IDRiD('./data/IDRiD_segmentation')
    for img_id, img, labels in dset.iter_imgs(labels=labels):
        bg = util.get_background(img)[:, :, 0]
        for lesion_name, mask in labels.items():
            yield (img_id, lesion_name, img, mask, bg)


class BayesDiseasedPixel(dict):
    """
    Bayesian Network to find the probability of a class label given pixel color

    p(label|r,g,b) = p(r,g,b|label)p(label) / p(r,g,b)
    \___________/    \___________/ \______/   \_____/
          |                |           |         |
       posterior       likelihood    prior    marginal

    """
    def __init__(self, labels, bins=256, compute_marginal=True):
        """
        labels - List[str] - list of labels (lesion names) considered
        bins - assume 256 pixels.
        compute_marginal - whether to find and use p(r,g,b)
        """
        for label in labels:
            self['p(rgb|%s,D)' % label] = np.zeros((bins, bins, bins))
            self['p(%s|D)' % label] = 0
            self['count_%s' % label] = 0

            # TODO: remove after sanity
            self['p(rgb|not(%s),D)' % label] = np.zeros((bins, bins, bins))
        if compute_marginal:
            self['p(rgb|D)'] = 0
        self.bins = bins
        self._set_of_observations = set()  # store the images considered to be
        #  able to compute p(rgb) without double counting images.  This is
        #  necessary since all images may not have masks for all labels
        self.compute_marginal = compute_marginal

    def _update_count(self, label_name):
        self['count_%s' % label_name] += 1

    def _update_prior(self, label_name, mask, bg):
        # update p(label|D).  assume p(I) = 1/|D|  (each image equal weight)
        k = 'p(%s|D)' % label_name
        p = mask.sum() / (~bg).sum()  # p(label|I)
        n = self['count_%s' % label_name]
        self[k] = (1-1/n)*self[k] + p/n

    def _update_likelihood_and_marginal(self, label_name, img, mask, bg,
                                        update_marginal=True):
        n = self['count_%s' % label_name]

        def _likelihood():
            H_diseased, _ = np.histogramdd(
                img[~bg & mask], bins=self.bins,
                range=[(0, 1), (0, 1), (0, 1)])
            k_diseased = 'p(rgb|%s,D)' % label_name
            p = H_diseased / H_diseased.sum()  # p(rgb|lesion,I,D)
            # update: p(rgb|lesion,I,D)p(I,D) = sum_I p(rgb|lesion,I,D)p(I)
            self[k_diseased] = (1-1/n)*self[k_diseased] + p/n
            assert np.allclose(self[k_diseased].sum(), 1)
            return H_diseased
        H_diseased = _likelihood()

        # TODO: after remove sanity, only run if update_marginal
        H_healthy, _ = np.histogramdd(
            img[~bg & ~mask], bins=self.bins, range=[(0, 1), (0, 1), (0, 1)])

        def _marginal():
            k_marginal = 'p(rgb|D)'
            H = H_diseased + H_healthy
            p = H / H.sum()  # p(rgb|I,D)
            self[k_marginal] = (1-1/n)*self[k_marginal] + p/n  # p(rgb|I,D)p(I|D)
            assert np.allclose(self[k_marginal].sum(), 1)
        if update_marginal:
            _marginal()

        def _likelihood_not_diseased():  # sanity checking
            # add healthy p(rgb|healthy,D) for sanity check?
            # TODO: remove this after sanity check done.
            k_healthy = 'p(rgb|not(%s),D)' % label_name
            p_healthy = H_healthy / H_healthy.sum()
            self[k_healthy] = (1-1/n)*self[k_healthy] + p_healthy/n
            assert np.allclose(self[k_healthy].sum(), 1)
        _likelihood_not_diseased()

    def update_stats(self, img_id, label_name, img, mask, bg):
        # center all images so rgb values are comparable.
        # TODO: rescale so the average is 0.5, without messing [0, 1] bounds.
        #  pixels += 0.5 - pixels.mean(axis=0, keepdims=True)

        self._update_count(label_name)
        is_new_img = img_id not in self._set_of_observations

        self._update_prior(label_name, mask, bg)
        self._update_likelihood_and_marginal(
            label_name, img, mask, bg,
            update_marginal=self.compute_marginal and is_new_img)
        self._set_of_observations.add(img_id)

    def eval_posterior(self, label_name, rgb):
        """
        Compute rhs of p(label|r,g,b) = p(r,g,b|label)p(label)/p(r,g,b)

        #  Compute rhs of p(label|r,g,b)p(r,g,b) = p(r,g,b|label)p(label)
        #  It is expensive to compute p(r,g,b) so we just dont compute it.

        label_name - str.  one of the labels the model was trained on.
        rgb - ndarray.  a Nx3 matrix of pixel values. columns are R,G,B.
        """
        # look up the index of the closest bin
        ri, gi, bi = np.searchsorted(
            np.linspace(0, 1, self.bins)[1:],
            rgb.ravel(), side='right').reshape(rgb.shape).T
        # compute the probability
        prior = self['p(%s|D)' % label_name]
        likelihood = self['p(rgb|%s,D)' % label_name][ri, gi, bi]
        marginal = self['p(rgb|D)'][ri, gi, bi]

        l1 = self['p(rgb|not(%s),D)' % label_name][ri, gi, bi]
        X2 = (1 - prior) * l1 / marginal

        # TODO: add this after verify no missing data on the training set.
        #  missing_data_mask = marginal == 0
        #  likelihood[missing_data_mask] = 0  # 0 probability for missing data
        #  marginal[missing_data_mask] = 1  # avoid spurious error
        X = prior * likelihood# / marginal
        #  assert (X <= 1).all()
        assert np.allclose(1, X2 + X)
        return X

    def save(self, fp):
        with open(fp, 'wb') as fout:
            pickle.dump(self, fout)

    @staticmethod
    def load(fp):
        with open(fp, 'rb') as fin:
            rv = pickle.load(fin)
        return rv


def train(labels):
    S = BayesDiseasedPixel(labels)
    for img_id, lesion_name, img, mask, bg in iter_imgs(labels):
        print(img_id, lesion_name)
        S.update_stats(img_id, lesion_name, img, mask, bg)
        #  print(S)
    S.save('./data/idrid_bayes_prior.pickle')
    return S


def load_pretrained(fp='./data/idrid_bayes_prior.pickle'):
    return BayesDiseasedPixel.load(fp)


def get_transmission_map(label_name, img, bg):
    rgb = img[~bg]
    tmp = s.eval_posterior(label_name, rgb)

    #  assert tmp.max() <= 1
    #  assert tmp.min() >= 0
    rv = np.zeros(img.shape[:2])
    rv[~bg] = tmp  # / tmp.max()
    return rv

if __name__ == "__main__":
    s = load_pretrained()
    labels = IDRiD.labels
    labels = ('HE', )
    #  s = train(labels)

    for img_id, lesion_name, img, mask, bg in iter_imgs(labels):
        t = get_transmission_map(lesion_name, img, bg)
        print(t[mask].mean())
        #  assert (t[mask] >0).all()  # model missed a pixel of training img.
    #  #      print(img.min(), img.max())
    #  #      print('===')
    #  #      print(img_id, lesion_name)
        #  plt.clf()

        #  plt.imshow(np.dstack([t,mask, np.zeros_like(t)]))
        #  plt.pause(0.01)
        #  f, (a, b) = plt.subplots(2, 1, num=1, sharex=True, sharey=True)
        #  a.imshow(img)
    #      img[~mask] = 0
    #      b.imshow(img)
    #      plt.pause(0.01)


    # TMP hack TODO remove.  faster experimenting.
    #  img = img[1000: -1000, 1000: -1000]
    #  bg = bg[1000:-1000, 1000:-1000]
    #  mask = mask[1000:-1000, 1000:-1000]

    #  H_healthy, _ = np.histogramdd(
        #  img[~bg & ~mask], bins=256, range=[(0, 1), (0, 1), (0, 1)])
    #  H = H_diseased + H_healthy

    #  p(rgb|diseased,I)

    #  print(img[~bg].mean())
    #  plt.figure(num=1).clf()
    #  f, ax = plt.subplots(num=1)
    #  ax.imshow(np.dstack([(H_diseased).sum(1), (H_healthy).sum(1), np.zeros((256, 256))]))
    #  plt.pause(0.001)
#  diseased = img[mask & ~bg]
#  healthy = img[~mask & ~bg]


#  0. illuminate all imgs in dset (optional, only if enhances separation between healthy and disease)
#  1. 3d (sparse) histogram for each img. red x green x blue.  p(r,g,b | I)
#  2. add 3d histograms together for all imgs. normalize to get dist of colors in dataset  p(r,g,b | D)
#  3. count healthy pixels in image p(healthy | I)  --> and in dataset p(healthy | D)
#     count disease pixels ...       p(disease  | I)                     p(disease  | D)
#  4. count r,g,b tuples for all healthy pixels and normalize to get p(r_D,g_D,b_D | healthy)
#     repeat for disease pixels
#  5. via Bayes rule, get prob of a healthy pixel or disease pixel.  p(healthy | r_D,g_D,b_D) = p(healthy) * p(r,g,b|healthy) / p(r,g,b)
#     ---> now we have a lookup table for every possible pixel combination in the dataset.
#     code check:  assert p(healthy|r,g,b) = 1 - p(disease|r,g,b) forall r,g,b.
#  6. use this to populate a grayscale depth map, t.  (look up prob of each pixel, x in I. in other words,  t[x] = p(disease|r_I(x), b_I(x), g_I(x)))
#     --> t values close to 1 will brighten the image.
#     --> will max probabilities be very small?  If so then A(1-t) will be off and need to normalize t: t / t.max().  May happen if img is unlike the dataset.
#  7. Set A=0.1 and using t, solve I=Jt+A(1-t) for J.
#     Option 2:  Set A=1 and t=1-t and I=1-I.  Solve for J.  Let's us work around the "A=0" problem.
#     --> are these equivalent if A=0 in the first case?

#  ... also try replace 6 with this:
#  6. populate t with a combination of healthy and disease:
#      t[x] = sqrt( p(healthy)*p(disease)*p(disease))  --> geometric average of the ones we are most uncertain of:  p(h)p(d) and the ones that are diseased: p(d)

#      t[x] = max(0, p(disease|I)-p(healthy|I))  effectively amplifies diseased pixels that are most unlike healthy ones, and completely ignores diseased pixels that are most likely healthy... not great
#      t[x] = p(healthy) - (p(healthy)*p(disease))  --> discount the healthy pixels that are most certainly healthy and generally make all pixels look more diseased.
