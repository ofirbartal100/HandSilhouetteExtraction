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
    def set_kernel_value_at_pos(kernel, row, col, value):
        if row < 0 or row >= kernel.shape[0]:
            return
        if col < 0 or col >= kernel.shape[1]:
            return
        kernel[row, col] = value

    p = source - target
    # flip y axis
    p *= torch.from_numpy(np.array([1.0, -1.0]))

    abs_x_is_bigger = True
    if np.abs(p[0]) > np.abs(p[1]):
        # normalize to x step size
        p *= 1 / np.abs(p[0])
        abs_x_is_bigger = True
    else:
        p *= 1 / np.abs(p[1])
        abs_x_is_bigger = False

    mid_point_vector = p * (kernel_size / 2)
    translation_vector = int((kernel_size) / 2) - mid_point_vector
    kernel_down = np.zeros((kernel_size, kernel_size))
    x = 0
    y = 0
    while np.abs(x) < kernel_size and np.abs(y) < kernel_size:
        row_pos = int(y + translation_vector[1])
        col_pos = int(x + translation_vector[0])
        if abs_x_is_bigger:
            set_kernel_value_at_pos(kernel_down, row_pos - 1, col_pos, 1)
            set_kernel_value_at_pos(kernel_down, row_pos, col_pos, -1)
            # set_kernel_value_at_pos(kernel_down, row_pos, col_pos, 0)
            # set_kernel_value_at_pos(kernel_down, row_pos + 1, col_pos, -1)
        else:
            set_kernel_value_at_pos(kernel_down, row_pos, col_pos - 1, 1)
            set_kernel_value_at_pos(kernel_down, row_pos, col_pos, -1)
            # set_kernel_value_at_pos(kernel_down, row_pos, col_pos, 0)
            # set_kernel_value_at_pos(kernel_down, row_pos, col_pos + 1, -1)

        y += p[1]
        x += p[0]

    kernel_up = np.zeros((kernel_size, kernel_size))
    x = 0
    y = 0
    while np.abs(x) < kernel_size and np.abs(y) < kernel_size:
        row_pos = int(y + translation_vector[1])
        col_pos = int(x + translation_vector[0])
        if abs_x_is_bigger:
            # set_kernel_value_at_pos(kernel_up, row_pos - 1, col_pos, -1)
            # set_kernel_value_at_pos(kernel_up, row_pos, col_pos, 0)
            set_kernel_value_at_pos(kernel_up, row_pos, col_pos, -1)
            set_kernel_value_at_pos(kernel_up, row_pos + 1, col_pos, 1)
        else:
            # set_kernel_value_at_pos(kernel_up, row_pos, col_pos - 1, -1)
            # set_kernel_value_at_pos(kernel_up, row_pos, col_pos, 0)
            set_kernel_value_at_pos(kernel_up, row_pos, col_pos, -1)
            set_kernel_value_at_pos(kernel_up, row_pos, col_pos + 1, 1)
        y += p[1]
        x += p[0]

    return np.flip(kernel_down, axis=0), np.flip(kernel_up, axis=0)


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
plt.imshow(img, cmap='gray')
for i in [1, 5, 9, 13, 17]:
    for j in range(3):
        plt.plot(landmarks[i+j:i+j+2, 0].numpy().squeeze(), landmarks[i+j:i+j+2, 1].numpy().squeeze(), 'ro-')
# plt.subplot(221), plt.hist(img.ravel(), 256, [0, 256])
# plt.imshow(img, cmap='gray')
plt.show()
plt.clf()

# filtered = cv2.filter2D(img,cv2.CV_8U,np.array([[-1,-1,1],[-1,1,0],[1,0,0]]))


# kernel_down,kernel_up = create_kernel(landmarks[7],landmarks[8],5)
#
# filtered = cv2.filter2D(img, cv2.CV_8U, kernel_down)
# plt.imshow(filtered, cmap='gray')
# plt.show()
# plt.clf()
# filtered = cv2.filter2D(img, cv2.CV_8U, kernel_up)
# plt.imshow(filtered, cmap='gray')
# plt.show()
# plt.clf()
# Smoothing without removing edges.
# gray_filtered = cv2.bilateralFilter(img,10 , 20 , 5)
#
# Z = img.reshape((-1,1))
#
# # convert to np.float32
# Z = np.float32(Z)
#
# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 8
# ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#
# # Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
# # Applying the canny filter
edges = cv2.Canny(img, 50, 200)
# # edges_filtered = cv2.Canny(gray_filtered, 30, 40)
#
plt.imshow(edges, cmap='gray')
plt.show()
plt.clf()
# plt.imshow(res2,cmap='gray')
# plt.show()


mask_size = 40
k_size = 5
all_masksed_images_combined = np.zeros(img.shape[:2], np.uint8)
# for i in range(landmarks.size(0)):
for i in [1, 5, 9, 13, 17]:
    # for i in [0]:
    for j in range(3):
        # hist_mask = create_hist_mask(img.shape[:2], landmarks[i+j], mask_size)
        # masked_img = cv2.bitwise_and(img, img, mask=hist_mask)
        hist_mask = create_hist_mask(img.shape[:2], landmarks[i + j], mask_size)
        masked_img = cv2.bitwise_and(img, img, mask=hist_mask)
        # edges = cv2.Canny(img, 100, 200)
        kernel_down, kernel_up = create_kernel(landmarks[i + j], landmarks[i + j + 1], k_size)
        filtered_down = cv2.filter2D(masked_img, cv2.CV_8U, kernel_down)

        # masked_edges = cv2.bitwise_and(edges, edges, mask=hist_mask)
        # kernel_down, kernel_up = create_kernel(landmarks[i+j], landmarks[i+j+1], k_size)
        # filtered_down = cv2.filter2D(masked_img, cv2.CV_8U, kernel_down)
        # filtered_up = cv2.filter2D(masked_img, cv2.CV_8U, kernel_up)
        # all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined,masked_img)
        # all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined,filtered_down)
        # all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined,filtered_up)
        # all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined,masked_edges)
        all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined, filtered_down)
        # masked_img_hist = cv2.calcHist([masked_img], [0], hist_mask, [256], [0, 256])
        # plt.subplot(222), plt.imshow(masked_img, cmap='gray')#, plt.scatter(landmarks[i, 0], landmarks[i, 1], c='red')
        # plt.subplot(221), plt.plot(masked_img_hist)
        # plt.show()
        # plt.clf()
plt.imshow(all_masksed_images_combined, cmap='gray')
for i in [1, 5, 9, 13, 17]:
    for j in range(3):
        plt.plot(landmarks[i+j:i+j+2, 0].numpy().squeeze(), landmarks[i+j:i+j+2, 1].numpy().squeeze(), 'ro-')
plt.show()
