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


def create_kernel(source,target,kernel_size):
    p = source - target
    # flip y axis
    p *= torch.from_numpy(np.array([1.0, -1.0]))
    if np.abs(p[0])>np.abs(p[1]):
        # normalize to x step size
        p *= 1 / np.abs(p[0])
    else:
        p *= 1 / np.abs(p[1])
    # get the edge y scatter, and plus 2 for above and under -1 2 -1
    # total_y = int(p[1]*kernel_size+2)
    total_y = int( np.abs(p[1])*kernel_size+1)
    total_x = int( np.abs(p[0])*kernel_size+1)

    content = np.zeros((total_y, kernel_size))
    kernel_down = np.zeros((kernel_size,kernel_size))
    x = 0
    y = 0
    if total_x>total_y:
        while x < kernel_size:
            # kernel[int(y)+int(total_y/2)-1, x] = -1
            # kernel[int(y)+int(total_y/2), x] = 2
            # kernel[int(y)+int(total_y/2)+1, x] = -1

            kernel_down[int(y) + int(total_y / 2) - 1, x] = -1
            kernel_down[int(y) + int(total_y / 2), x] = 1
            y += p[1]
            x += 1
    else:
        while y < kernel_size:
            # kernel[int(y)+int(total_y/2)-1, x] = -1
            # kernel[int(y)+int(total_y/2), x] = 2
            # kernel[int(y)+int(total_y/2)+1, x] = -1

            kernel_down[y,int(x) + int(total_x / 2) - 1] = -1
            kernel_down[y,int(x) + int(total_x / 2)] = 1
            x += p[0]
            y += 1

    kernel_up = np.zeros((kernel_size, kernel_size))
    x = 0
    y = 0
    if total_x > total_y:
        while x < kernel_size:
            # kernel[int(y)+int(total_y/2)-1, x] = -1
            # kernel[int(y)+int(total_y/2), x] = 2
            # kernel[int(y)+int(total_y/2)+1, x] = -1

            kernel_up[int(y) + int(total_y / 2) + 1, x] = -1
            kernel_up[int(y) + int(total_y / 2), x] = 1
            y += p[1]
            x += 1
    else:
        while y < kernel_size:
            # kernel[int(y)+int(total_y/2)-1, x] = -1
            # kernel[int(y)+int(total_y/2), x] = 2
            # kernel[int(y)+int(total_y/2)+1, x] = -1

            kernel_up[y, int(x) + int(total_x / 2) + 1] = -1
            kernel_up[y, int(x) + int(total_x / 2)] = 1
            x += p[0]
            y += 1
    return np.flip(kernel_down,axis=0),np.flip(kernel_up,axis=0)



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
# plt.subplot(222), plt.imshow(img, cmap='gray'), plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red')
# plt.subplot(221), plt.hist(img.ravel(), 256, [0, 256])
# plt.imshow(img, cmap='gray')
# plt.show()
# plt.clf()

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
plt.imshow(edges,cmap='gray')
plt.show()
plt.clf()
# plt.imshow(res2,cmap='gray')
# plt.show()



mask_size = 40
k_size=5
all_masksed_images_combined = np.zeros(img.shape[:2],np.uint8)
# for i in range(landmarks.size(0)):
for i in [1,5,9,13,17]:
# for i in [0]:
    for j in range(3):
        # hist_mask = create_hist_mask(img.shape[:2], landmarks[i+j], mask_size)
        # masked_img = cv2.bitwise_and(img, img, mask=hist_mask)
        hist_mask = create_hist_mask(img.shape[:2], landmarks[i + j], mask_size)
        masked_img = cv2.bitwise_and(img, img, mask=hist_mask)
        edges = cv2.Canny(img, 100, 200)
        masked_edges = cv2.bitwise_and(edges, edges, mask=hist_mask)
        # kernel_down, kernel_up = create_kernel(landmarks[i+j], landmarks[i+j+1], k_size)
        # filtered_down = cv2.filter2D(masked_img, cv2.CV_8U, kernel_down)
        # filtered_up = cv2.filter2D(masked_img, cv2.CV_8U, kernel_up)
        # all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined,masked_img)
        # all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined,filtered_down)
        # all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined,filtered_up)
        all_masksed_images_combined = cv2.bitwise_or(all_masksed_images_combined,masked_edges)
        # masked_img_hist = cv2.calcHist([masked_img], [0], hist_mask, [256], [0, 256])
        # plt.subplot(222), plt.imshow(masked_img, cmap='gray')#, plt.scatter(landmarks[i, 0], landmarks[i, 1], c='red')
        # plt.subplot(221), plt.plot(masked_img_hist)
        # plt.show()
        # plt.clf()
plt.imshow(all_masksed_images_combined,cmap='gray')
plt.show()
