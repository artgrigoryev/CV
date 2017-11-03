# CV classes
# Assignment 1 
# Grigoryev Artyom

from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

def histeq(im,nbr_bins=256):
    """ Histogram equalization of a grayscale image. """ 
    # get image histogram 
    imhist,bins = np.histogram(im.flatten(), nbr_bins, normed=True) 
    cdf = imhist.cumsum() 
    # cumulative distribution function 
    cdf = 255 * cdf / cdf[-1] 
    # normalize 
    # use linear interpolation of cdf to find new pixel values 
    im2 = np.interp(im.flatten(),bins[:-1],cdf) 
    return im2.reshape(im.shape), cdf 


def GetFigSubplots():
    ax = []
    ax.append(plt.subplot2grid((4,2), (0,0)))
    ax.append(plt.subplot2grid((4,2), (0,1)))
    ax.append(plt.subplot2grid((4,2), (1,0)))
    ax.append(plt.subplot2grid((4,2), (1,1)))
    ax.append(plt.subplot2grid((4,2), (2,0), colspan = 2))
    ax.append(plt.subplot2grid((4,2), (3,0)))
    ax.append(plt.subplot2grid((4,2), (3,1)))
    return ax


def TaskLecture3():
    im = np.array(Image.open('./Images/lena_gray.gif').convert('L')) 
    im2, im_cdf = histeq(im)

    norm_vals = np.random.normal(127, 32, im.shape)
    norm_vals = norm_vals.clip(0,255)

    ref_hist,bins = np.histogram(norm_vals.flatten(), 256, normed=True)
    ref_cdf = ref_hist.cumsum()
    ref_cdf = 255*(ref_cdf/ref_cdf[-1])
    res_im = np.reshape(np.interp(im2.flatten(), ref_cdf, bins[:-1]), im.shape)

    fig = plt.figure()
    plt.gray()

    ax = GetFigSubplots()
    ax[0].imshow(im)
    ax[0].axis('off')
    
    ax[1].imshow(res_im)
    ax[1].axis('off')
    
    ax[2].hist(im.flatten(),128)
    ax[2].axes.get_xaxis().set_visible(False)
    ax[2].axes.get_yaxis().set_visible(False)

    ax[3].hist(res_im.flatten(),128)
    ax[3].axes.get_xaxis().set_visible(False)
    ax[3].axes.get_yaxis().set_visible(False)

    ax[4].hist(norm_vals.flatten(),128)
    ax[4].axes.get_xaxis().set_visible(False)
    ax[4].axes.get_yaxis().set_visible(False)

    ax[5].plot(im_cdf)
    ax[5].axis('on')

    ax[6].plot(ref_cdf)
    ax[6].axis('on')
    plt.savefig('./Images/TaskLecture3.jpg', dpi = 1200)
    plt.show()
    
if __name__ == '__main__':
    TaskLecture3()