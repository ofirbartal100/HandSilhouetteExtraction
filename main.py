import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def extract_mask_location(point, mask_size):
    return int(point[0] - mask_size / 2), int(point[0] + mask_size / 2), int(point[1] - mask_size / 2), int(
        point[1] + mask_size / 2)


def create_hist_mask(shape, point, mask_size):
    mask = np.zeros(shape, np.uint8)
    x_range_start, x_range_stop, y_range_start, y_range_stop = extract_mask_location(point, mask_size)
    mask[y_range_start:y_range_stop, x_range_start:x_range_stop] = 255
    return mask


# load and extract data

images_path = r"D:\TAU\Research\Project\Dataset\GANeratedDataset_v3\GANeratedHands_Release\data\noObject\\"
csv_path = r"D:\TAU\Research\Project\Learning\Results\LandmarksExtraction\raw_dataframes_clean_fromt_hand.csv"

with open(csv_path, 'r') as csv:
    line = csv.readlines()[57]
    line_values = line.split(',')

image_name = line_values[0].split('_', 1)
full_image_path = images_path + image_name[0] + '\\' + image_name[1]
landmarks = np.array(line_values[1:], dtype=float)
landmarks = torch.from_numpy(landmarks).view(-1, 2)

# plot image , landmarks and full histogram

img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
plt.subplot(222), plt.imshow(img, cmap='gray'), plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red')
plt.subplot(221), plt.hist(img.ravel(), 256, [0, 256])
plt.show()
plt.clf()

mask_size = 20
for i in range(landmarks.size(0)):
# for i in [0]:
    hist_mask = create_hist_mask(img.shape[:2], landmarks[i], mask_size)
    masked_img = cv2.bitwise_and(img, img, mask=hist_mask)
    masked_img_hist = cv2.calcHist([masked_img], [0], hist_mask, [256], [0, 256])
    plt.subplot(222), plt.imshow(masked_img, cmap='gray')#, plt.scatter(landmarks[i, 0], landmarks[i, 1], c='red')
    plt.subplot(221), plt.plot(masked_img_hist)
    plt.show()
    plt.clf()

