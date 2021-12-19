import cv2
import os
import numpy as np
import torch
import math
## get image paths

IMG_EXTENSIONS = [".jpg",".JPG",".jpeg",".JPEG",".png",".PNG",".ppm",".PPM",".bmp",".BMP",".tif"]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dataroot):
    paths = None
    if isinstance(dataroot, str):
        paths = sorted(get_paths_from_images(dataroot))
    elif isinstance(dataroot,list):
        paths = []
        for i in dataroot:
            paths += sorted(get_paths_from_images(i))
    return paths

def get_paths_from_images(path):
    assert os.path.isdir(path), "{:s} is not a valid directory".format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath,fname)
                images.append(img_path)
    assert images, "{:s} has no valid image file".format(path)
    return images

def single2tensor3(img):
    # opencv shape >>>> torch shape (HXWXC)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2,0,1).float()

def uint2stand(img):
    return np.float32(img/255.)

def imread_uint(img_paths):
    img = cv2.imread(img_paths, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def modcrop(img_in, scale):
    # img : np
    img = np.copy(img_in)
    H, W, C = img.shape
    H_r, W_r = H % scale, W % scale
    img = img[:H - H_r, :W - W_r, :]

    return img

def augment_img (img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.rot90(img, k=3)
    elif mode == 3:
        return np.rot90(img,k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img,k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img,k=3))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# convert 2/3/4 - dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0,1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1,2,0))
    return np.uint8((img*255.0).round())

def imsave(img, path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:,:,[2,1,0]]
    cv2.imwrite(path, img)

def calculate_psnr(pred, org, border=0):
    if not pred.shape == org.shape:
        raise ValueError("Input images must have the same dimensions")
    h, w = pred.shape[:2]
    pred = pred[border:h-border, border:w-border]
    org = org[border:h-border, border:w-border]

    pred = pred.astype(np.float64)
    org = org.astype(np.float64)

    mse = np.mean((pred-org)**2)
    if mse == 0:
        return float("inf")

    return 20 * math.log10(255.0 / math.sqrt(mse))