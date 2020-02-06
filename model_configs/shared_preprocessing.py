import scipy.ndimage as ndi
import numpy as np
import ietk
import torch


def pil_to_numpy(pil_img):
    return np.array(pil_img)


def preprocess(img_mask_tensor, method_name,
               resize_to=(512, 512), crop_to_size=(512, 512),
               **affine_transform_kws):
    """
    For retinal fundus images, with or without an associated image segmentation mask.
    center crop, resize to 1600x1600, randomly rotate +-30degrees and flip ud
    and lr.  Return the image and a focus region mask.

    input_img can have any number of channels.  Useful for semantic
    segmentation if you wish to stack the image and label masks together and
    then apply transforms.

    Assume input_mask_tensor is a numpy array and the first three channels
    is a [0,255] normalized image.
    """
    im = img_mask_tensor[:,:,:3]
    label_mask = img_mask_tensor[:,:,3:]
    h, w, ch = im.shape
    # --> get forground and center crop to minimize background
    im, fg, label_mask = ietk.util.center_crop_and_get_foreground_mask(
        im, is_01_normalized=False, label_img=label_mask)
    # --> enhance via ietk method A+B+X or whatever
    im = ietk.methods.all_methods[method_name](im/255, focus_region=fg)
    im = im.clip(0,1) * 255
    # --> affine transform
    stack = np.dstack([im, label_mask])
    stack = affine_transform(stack, size=resize_to, **affine_transform_kws)
    # --> random crop
    if resize_to != crop_to_size:
        stack = random_crop(stack, crop_to_size)
    # --> undo the stack and restore proper types, normalized in [0,1]
    im, label_mask = (stack[:,:,:ch]/255).astype('float32'), stack[:,:,ch:].astype(bool)
    im = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    label_mask = torch.tensor(label_mask, dtype=torch.float32).permute(2,0,1)
    return im, label_mask


def affine_transform(im, size=(512,512), **kws):
    """
    assume a square input image with any num of channels
    By default,
        rescale and crop to (512,512)
        random flipping in x and y
        random rotate -30 to 30 degrees
    Pass extra keyword args to change defaults;
        rot=int, flip_y=bool, flip_x=bool
    """
    h, w, ch = im.shape
    M1 = get_scale_flip_matrix(
        h=h, w=w, scale_y=h/size[0], scale_x=w/size[1],
        flip_y=kws.get('flip_y', np.random.randint(0,2))*(-2)+1,
        flip_x=kws.get('flip_x', np.random.randint(0,2))*(-2)+1)
    M2 = get_rotation_matrix(
        rot_degrees=kws.get('rot', np.random.uniform(-30, 30)), offset_h=size[0]/2, offset_w=size[0]/2)
    im, ma = im[:,:,:3], im[:,:,3:]
    ch = 3
    z = ndi.affine_transform(
        im, M1@M2, output_shape=(size[0], size[1], ch), prefilter=True, order=3)
    ch = ma.shape[-1]
    z2 = ndi.affine_transform(
        ma, M1@M2, output_shape=(size[0], size[1], ch), prefilter=False, order=0)
    z = np.dstack([z,z2])
    return z


def get_scale_flip_matrix(h, w, scale_y, scale_x, flip_y, flip_x):
    M = np.array([
        [flip_y * scale_y, 0, 0, w if flip_y==-1 else 0],
        [0, flip_x * scale_x, 0, h if flip_x==-1 else 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    return M


def get_rotation_matrix(rot_degrees, offset_h, offset_w):
    ang = rot_degrees * np.pi/180
    M_to_center = np.array([
        [1,0,0,offset_h],
        [0,1,0,offset_w],
        [0,0,1,0],
        [0,0,0,1] ])
    M = np.array([
        [np.cos(ang), -1.0*np.sin(ang), 0, 0],
        [np.sin(ang), np.cos(ang), 0, 0],
        [0, 0, 1, 0],
        [0,0,0,1]
        ])
    M_from_center = np.array([
        [1,0,0,-offset_h],
        [0,1,0,-offset_w],
        [0,0,1,0],
        [0,0,0,1] ])
    return M_to_center@M@M_from_center


def random_crop(im, size):
    y = np.random.randint(0, im.shape[0]-size[0]+1)
    x = np.random.randint(0, im.shape[1]-size[1]+1)
    return im[y:y+size[0],x:x+size[0]]


def cutout_inplace(im4, pct_side=.2):
    ch, h, w = im4.shape
    dy, dx = int(pct_side*h), int(pct_side*w)
    y = torch.randint(0, h-dy, (1,))
    x = torch.randint(0, w-dx, (1,))
    im4[y:min(h,y+dy),x:min(w,x+dx)] = 0
    return im4
