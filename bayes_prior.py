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


class Stats(dict):
    """
    Bayesian Network for inferring the probability of disease given pixel color
    """
    def __init__(self, labels=('MA', 'HE', 'SE', 'EX', 'OD'), bins=256):
        """
        labels - List[str] - list of labels (lesion names) considered
        """
        for label in labels:
            self['p(rgb|%s,D)' % label] = np.zeros((bins, bins, bins))
            self['p(%s|D)' % label] = 0
        self['count'] = 0
        self.bins = bins

    def _update_count(self):
        self['count'] += 1

    def _update_prior(self, label_name, mask, bg):
        # update p(label|D).  assume p(I) = 1/|D|  (each image equal weight)
        k = 'p(%s|D)' % label_name
        p = mask.sum() / (~bg).sum()  # p(label|I)
        n = self['count']
        self[k] = (1-1/n)*self[k] + p/n
        print('prior', self[k])

    def _update_likelihood(self, label_name, img, mask, bg):
        k = 'p(rgb|%s,D)' % label_name
        # center all images so rgb values are comparable.
        pixels = img[~bg & mask]
        # TODO: rescale so the average is 0.5, without messing [0, 1] bounds.
        #  pixels += 0.5 - pixels.mean(axis=0, keepdims=True)

        # get counts, convert to probabilities, update parameters
        H_diseased, _ = np.histogramdd(
            pixels, bins=self.bins, range=[(0, 1), (0, 1), (0, 1)])
        n = self['count']
        p = H_diseased / H_diseased.sum()
        self[k] = (1-1/n)*self[k] + p/n
        print('likelihood', self[k][self[k]!=0].mean())

    def update_stats(self, label_name, img, mask, bg):
        self._update_count()
        self._update_prior(label_name, mask, bg)
        self._update_likelihood(label_name, img, mask, bg)

    def eval_posterior(self, label_name, r, g, b):
        """
        Compute rhs of p(label|r,g,b)p(r,g,b) = p(r,g,b|label)p(label)

        Dont forget to center your image before picking the rgb value!
            >>> centered_img = img - img.mean((0, 1)) + .5

        It is expensive to compute p(r,g,b) so we just dont compute it.
        """
        # look up the index of the closest bin
        ri, gi, bi = np.digitize([r,g,b], np.linspace(0, 1, self.bins+1)[1:])
        print(ri, gi, bi)
        # compute the probability
        prior = self['p(%s|D)' % label_name]
        likelihood = self['p(rgb|%s,D)' % label_name][ri, gi, bi]
        return prior * likelihood

    def save(self, fp):
        with open(fp, 'wb') as fout:
            pickle.dump(self, fout)

    @staticmethod
    def load(fp):
        with open(fp, 'rb') as fin:
            rv = pickle.load(fin)
        return rv


def train():
    labels = IDRiD.labels
    S = Stats(labels)
    for img_id, lesion_name, img, mask, bg in iter_imgs(labels):
        print(img_id, lesion_name)
        S.update_stats(lesion_name, img, mask, bg)
        #  print(S)
    S.save('./data/idrid_bayes_prior.pickle')
    return S


def load_pretrained(fp='./data/idrid_bayes_prior.pickle'):
    return Stats.load(fp)

if __name__ == "__main__":
    #  s = load_pretrained()
    s = train()

    #  for img_id, lesion_name, img, mask, bg in iter_imgs(('MA',)):
        #  print(img.min(), img.max())
        #  print('===')
        #  print(img_id, lesion_name)
        #  plt.figure(num=1)
        #  plt.clf()
        #  plt.figure(num=1)
        #  plt.imshow(img)
        #  plt.pause(0.01)


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
