import cv2
import numpy as np
from PIL import Image


def cv2pil(cvimg):
    return Image.fromarray(cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB))


def imread(path, tw, th):
    img = cv2.imread(path)
    new_w, new_h = [int(wh*min(tw/img.shape[1], th/img.shape[0])) for wh in (img.shape[1], img.shape[0])]
    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    canvas[(th-new_h)//2:(th-new_h)//2+new_h, (tw-new_w)//2:(tw-new_w)+new_w] = cv2.resize(img, (new_w, new_h))
    return canvas, new_w, new_h


if __name__ == '__main__':
    pass
