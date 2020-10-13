import cv2
import matplotlib.pyplot as plt

def is_gray_scale(image):
    if len(image.shape) == 2:
        return True
    if len(image.shape) == 3 and image.shape[2] == 1:
        return True
    return False

def plt_show(image, figsize = None, is_axis_shown = True):
    if figsize != None:
        plt.figure(figsize = figsize)
    if not is_axis_shown:
        plt.xticks([])
        plt.yticks([])
    if is_gray_scale(image):
        plt.imshow(image, cmap = 'gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap = None)
    plt.show()