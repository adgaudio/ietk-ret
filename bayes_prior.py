"""
illumination with bayesian prior  - use the equation to increase brightness of
healthy pixels and decrease disease pixels.  this is possible since the depth
map can be thought of as probabilities.
"""
import numpy as np
from matplotlib import pyplot as plt
import dill as pickle
import functools


from sharpen_img import sharpen
from idrid import IDRiD
import util


def iter_imgs(labels):
    dset = IDRiD('./data/IDRiD_segmentation')
    for img_id, img, labels in dset.iter_imgs(labels=labels):
        fg = util.get_foreground(img)[:, :, 0]

        # center all images so rgb values are comparable.
        img = util.zero_mean(img, fg)

        for lesion_name, mask in labels.items():
            yield (img_id, lesion_name, img, mask, fg)


class BayesDiseasedPixel(dict):
    """
    Bayesian Network to find the probability of a class label given pixel color

    p(label|r,g,b) = p(r,g,b|label)p(label) / p(r,g,b)
    \___________/    \___________/ \______/   \_____/
          |                |           |         |
       posterior       likelihood    prior    marginal


    Implementation note: due to numerical precision issues, some probabilities
    are stored as counts.  Also, if the probabilities of one class are very
    small, this class figures this out and during inference returns the
    probability computed from the majority class via:
        min( p(rgb|label), 1 - p(rgb|not(label)) )
    """
    def __init__(self, labels, bins=256):
        """
        labels - List[str] - list of labels (lesion names) considered
        bins - assume 256 pixels.
        """
        for label in labels:
            self['H(rgb|%s,D)' % label] = np.zeros((bins, bins, bins))
            self['p(%s|D)' % label] = 0
            self['count_%s' % label] = 0

            self['H(rgb|not(%s),D)' % label] = np.zeros((bins, bins, bins))
            self['H(rgb|D)'] = 0
        self.bins = bins
        self._set_of_observations = set()  # store the images considered to be
        #  able to compute p(rgb) without double counting images.  This is
        #  necessary since all images may not have masks for all labels

    def _update_count(self, label_name):
        self['count_%s' % label_name] += 1

    def _update_prior(self, label_name, mask, fg):
        k = 'p(%s|D)' % label_name
        p = mask.sum() / (fg).sum()  # p(label|I)
        n = self['count_%s' % label_name]
        self[k] = (1-1/n)*self[k] + p/n  # p(label|D) = \sum p(label|I,D)p(I|D)

    def _update_likelihood_and_marginal(self, label_name, img, mask, fg,
                                        update_marginal=True):
        #  n = self['count_%s' % label_name]

        def _likelihood():
            H_diseased, _ = np.histogramdd(
                img[fg & mask], bins=self.bins,
                range=[(0, 1), (0, 1), (0, 1)])
            k_diseased = 'H(rgb|%s,D)' % label_name
            #  p = H_diseased / H_diseased.sum()  # p(rgb|lesion,I,D)
            #  self[k_diseased] = (1-1/n)*self[k_diseased] + p/n
            self[k_diseased] += H_diseased
            return H_diseased

        def _likelihood_not_diseased():
            H_healthy, _ = np.histogramdd(
                img[fg & ~mask],
                bins=self.bins, range=[(0, 1), (0, 1), (0, 1)])
            k_healthy = 'H(rgb|not(%s),D)' % label_name
            #  p_healthy = H_healthy / H_healthy.sum()
            #  self[k_healthy] = (1-1/n)*self[k_healthy] + p_healthy/n
            self[k_healthy] += H_healthy
            return H_healthy

        def _marginal():
            k_marginal = 'H(rgb|D)'
            H = H_diseased + H_healthy
            #  p = H / H.sum()  # p(rgb|I,D)
            #  self[k_marginal] = (1-1/n)*self[k_marginal] + p/n
            self[k_marginal] += H

        H_healthy = _likelihood_not_diseased()
        H_diseased = _likelihood()
        if update_marginal:
            _marginal()

    def update_stats(self, img_id, label_name, img, mask, fg):
        self._update_count(label_name)
        is_new_img = img_id not in self._set_of_observations

        self._update_prior(label_name, mask, fg)
        self._update_likelihood_and_marginal(
            label_name, img, mask, fg,
            update_marginal=is_new_img)
        self._set_of_observations.add(img_id)

    def eval_posterior(self, label_name, rgb):
        """
        Compute p(label|rgb) = p(label,rgb) / p(rgb)
        """
        # look up the index of the closest bin
        ri, gi, bi = np.searchsorted(
            np.linspace(0, 1, self.bins)[1:],
            rgb.ravel(), side='right').reshape(rgb.shape).T
        # get probability
        x = self['H(rgb|%s,D)' % label_name][ri, gi, bi]
        z = self['H(rgb|D)'][ri, gi, bi]
        X = x / z  # p(rgb,MA) / p(rgb)
        # sanity check
        #  _nk_li = 'H(rgb|not(%s),D)' % label_name
        #  y = self[_nk_li][ri, gi, bi]
        #  X2 = y / (x+y)  # p(rgb,not(MA)) / p(rgb)
        #  assert np.allclose(self[_k_li] + self[_nk_li], self['H(rgb|D)'])
        #  assert X.min() >= 0
        #  assert X.max() <= 1
        #  assert X2.min() >= 0
        #  assert X2.max() <= 1
        #  assert (X + X2 == 1).all()
        return X


def train(labels):
    S = BayesDiseasedPixel(labels)
    for img_id, lesion_name, img, mask, fg in iter_imgs(labels):
        print(img_id, lesion_name)
        S.update_stats(img_id, lesion_name, img, mask, fg)
    # save model
    with open('./data/idrid_bayes_prior.pickle', 'wb') as fout:
        fout.write(pickle.dumps(S))
    return S


@functools.lru_cache()
def load_pretrained(fp='./data/idrid_bayes_prior.pickle'):
    with open(fp, 'rb') as fin:
        rv = pickle.loads(fin.read())
    # do weird things to work around annoying pickle issues:
    S = BayesDiseasedPixel(())
    S.update(rv)
    S.__dict__.update(rv.__dict__)
    return S


def get_transmission_map(label_name, img, focus_region, model=None):
    if model is None:
        model = load_pretrained()
    rgb = img[focus_region]
    tmp = model.eval_posterior(label_name, rgb)
    rv = np.zeros(img.shape[:2])
    rv[focus_region] = tmp
    return rv


def bayes_sharpen(img, label_name, focus_region):
    t = get_transmission_map(label_name, img, focus_region)
    print(t.min(), t.max())
    t = 1 - t
    z = t[focus_region]
    desired_range = .3
    desired_min = 0.001
    t = (t - z.min()) / (z.max() - z.min()) * desired_range + desired_min
    z = t[focus_region]
    print(z.min(), z.max())
    #  t = t.clip(.15, 1)
    bg = ~focus_region
    t[bg] = 0
    sharp = sharpen(img, bg, t=t)
    return sharp


if __name__ == "__main__":
    labels = IDRiD.labels
    #  labels = ['HE']
    #  s = load_pretrained()
    s = train(labels)
    print('done train')

    for img_id, lesion_name, img, mask, fg in iter_imgs(labels):

        plt.figure(num=1)
        plt.clf()
        f, (a,b,c) = plt.subplots(1, 3, num=1, figsize=(12, 12))
        f.suptitle('%s %s' % (img_id, lesion_name))

        a.imshow(sharpen(img, ~fg, t=.15))
        b.imshow(bayes_sharpen(img, lesion_name, fg))

        t = get_transmission_map(lesion_name, img, fg)
        c.imshow(np.dstack([t,mask, t>0]))
        plt.pause(0.01)
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
