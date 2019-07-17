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


def create_kernel(source, target, kernel_size):
    direction = target - source
    direction *= torch.from_numpy(np.array([1.0, -1.0]))
    direction *= 1/direction.norm()
    sobely = torch.from_numpy(np.array([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]
    ]))

    # sobely = torch.from_numpy(np.array([
    #     [2.,1., 0., -1.,-2.],
    #     [2.,1., 0., -1.,-2.],
    #     [4.,2., 0., -2.,-4.],
    #     [2.,1., 0., -1.,-2.],
    #     [2.,1., 0., -1.,-2.],
    # ]))

    # sobely = torch.from_numpy(np.array([
    #     [1., 2., 0., -2., -1.],
    #     [2., 3., 0., -3., -2.],
    #     [3., 5., 0., -5., -3.],
    #     [2., 3., 0., -3., -2.],
    #     [1., 2., 0., -2., -1.],
    # ]))

    sobelx = sobely.t()

    direction_filter = sobelx * direction[0] + sobely * direction[1]
    return direction_filter.numpy(),direction_filter.numpy()*-1



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

img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
for i in [1, 5, 9, 13, 17]:
    for j in range(3):
        plt.plot(landmarks[i + j:i + j + 2, 0].numpy().squeeze(), landmarks[i + j:i + j + 2, 1].numpy().squeeze(),
                 'ro-')
plt.show()
plt.clf()

edges = cv2.Canny(img, 50, 200)
plt.imshow(edges, cmap='gray')
plt.show()
plt.clf()


mask_size = 40
k_size = 5
all_masksed_images_combined = np.zeros(img.shape[:2], np.uint8)
for i in [1, 5, 9, 13, 17]:
    for j in range(3):
        hist_mask = create_hist_mask(img.shape[:2], landmarks[i + j], mask_size)
        masked_img = cv2.bitwise_and(img, img, mask=hist_mask)

        kernel_down, kernel_up = create_kernel(landmarks[i + j], landmarks[i + j + 1], k_size)

        filtered_down = cv2.filter2D(img, cv2.CV_8U, kernel_down)
        filtered_up = cv2.filter2D(img, cv2.CV_8U, kernel_up)
        filtered_down = cv2.bitwise_and(filtered_down, filtered_down, mask=hist_mask)
        filtered_up = cv2.bitwise_and(filtered_up, filtered_up, mask=hist_mask)

        all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined, filtered_down)
        all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined, filtered_up)


plt.imshow(all_masksed_images_combined, cmap='gray')

plt.show()
