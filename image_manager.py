import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import glob



def set_of_images_location(path):

    directory= os.listdir(path)

    p = []
    for d in directory:
        source_path  = os.path.join(path, d)
        FN = glob.glob(os.path.join(source_path,'*.jpg'))
        p.append(FN)

    return p



def image_read(image_location):
    M = cv.imread(image_location)
    if M is None:
        raise IOError('cannot find immage %s' % image_location)

    return M



def image_show(M,figsize=(8, 8), gray_image=False):
    """
    show image using a gray scale

    image_show(M)

    Parameters
    ----------

    M numpy matrix of the image
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    if gray_image:
      ax.imshow(M,'gray')
    else:
      ax.imshow(M)

    ax.grid(False)


    
def spots_detect_for_prediction(M, threshold=180, s_min=100, s_max=500000, blur=(30,30)):

    M1 = cv.cvtColor         (M, cv.COLOR_BGR2GRAY)
    M2 = cv.adaptiveThreshold(M1, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    M3 = cv.blur             (M2, blur)
    M4 = cv.threshold        (M3, 180,255,cv.THRESH_BINARY_INV)[-1]
    contours, hierarchy = cv.findContours(M4, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    contours_selected   = [ ct for ct in contours if s_min <= cv.contourArea(ct) <= s_max]

    M5 = np.zeros_like(M4)
    cv.drawContours(M5, contours, -1, (255,0,0), 30)
    cv.drawContours(M5, contours, -1, (255,0,0), -50)
    M6 = cv.resize(M5,(150,150))
    M7 = cv.threshold (M6, 1,255,cv.THRESH_BINARY)[-1]
    M8 = cv.cvtColor(M7, cv.COLOR_GRAY2RGB)
    M9 = np.expand_dims(M8, axis = 0)

    return M9, len(contours_selected), contours_selected
