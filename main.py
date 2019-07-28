import processedImage

plotter = processedImage.ProcessedImagePlotter()

image_num = [57, 100, 200, 600, 8000, 2547, 6004]
p_images = [processedImage.ProcessedImage(i) for i in image_num]

transforms = [
    processedImage.SteeredEdgeTransform(3, 40, False),
    processedImage.CannyTransform(100, 200, True)
]

# plotter.plot_multy_grid(p_images, transforms)

#
# import cv2 as cv
# import numpy as np
# import argparse
# import random as rng
# rng.seed(12345)
#
#
# def thresh_callback(val):
#     threshold = val
#     # Detect edges using Canny
#     canny_output = cv.Canny(src_gray, threshold, threshold * 2)
#     # Find contours
#     _, contours, hierarchy = cv.findContours(canny_output, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
#     # Draw contours
#     drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
#     for i in range(len(contours)):
#         color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
#         cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
#     # Show in a window
#     cv.imshow('Contours', drawing)
#
# # Load source image
# # Convert image to gray and blur it
# src_gray = transforms[0].transform(p_images[0])
# src_gray = cv.blur(src_gray, (3,3))
# # Create Window
# source_window = 'Source'
# cv.namedWindow(source_window)
# cv.imshow(source_window, transforms[0].transform(p_images[0]))
# max_thresh = 255
# thresh = 100 # initial threshold
# cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
# thresh_callback(thresh)
# cv.waitKey()


import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# img = transforms[0].transform(p_images[0])
img = p_images[0].img
# i = 3
# # intervals
# s = np.linspace(0, 2 * np.pi, 400)
# # center
# c = (p_images[0].landmarks[i] + p_images[0].landmarks[i + 1]).numpy() / 2
# # radius of ellipse
# r = (p_images[0].landmarks[i + 1] - p_images[0].landmarks[i])
# # cos ,sin
# d2 = r / r.norm()
# cos_theta = d2[0].item()
# sin_theta = d2[1].item()
# rx = r.norm().item()
# ry = 15
# x = rx * np.cos(s) * cos_theta - ry * np.sin(s) * sin_theta + c[0]
# y = rx * np.cos(s) * sin_theta + ry * np.sin(s) * cos_theta + c[1]
x = np.array([12, 21, 53, 70, 62
                 , 100, 148, 165, 156, 114
                 , 128, 163, 188, 176, 146
                 , 148, 182, 200, 185, 144,
              159, 210, 225, 198, 124, 70, 25,
              12])
y = np.array([133, 86, 50, 70, 110,
              84, 61, 65, 82, 110,
              111, 88, 84, 108, 126,
              132, 109, 110, 135, 166,
              171, 170, 182, 200, 213, 236,200
              ,133])
init = np.array([x, y]).T
snake = active_contour(gaussian(img, 3),
                       init, alpha=0.025, beta=10, gamma=0.001)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()
