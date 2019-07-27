import numpy as np
import torch
import cv2
import abc
import matplotlib.pyplot as plt


class ProcessedImagePlotter(object):
    def __init__(self, save_flag=False):
        self.save_flag = save_flag
        pass

    def plot_single(self, processedImage, transform=None):
        plt.clf()
        if transform:
            transformed_image = transform.transform(processedImage)
            plt.imshow(transformed_image, cmap='gray')
        else:
            plt.imshow(processedImage.img, cmap='gray')
        plt.show()

    def plot_single_grid(self, processedImage, transforms):

        if len(transforms) == 1:
            return self.plot_single(processedImage, *transforms)

        plt.clf()
        fig, axes = plt.subplots(len(transforms))
        i = 0
        for t in transforms:
            transformed_image = t.transform(processedImage)
            axes[i].imshow(transformed_image, cmap='gray')
            i = i + 1

        if self.save_flag:
            plt.savefig("transformed_single_grid", type="jpeg")
        plt.show()

    def plot_multy_grid(self, processedImages, transforms):
        plt.clf()
        fig, axes = plt.subplots(len(transforms), len(processedImages))
        i = 0
        for t in transforms:
            j = 0
            for pi in processedImages:
                transformed_image = t.transform(pi)
                if len(transforms) == 1:
                    axes[j].imshow(transformed_image, cmap='gray')
                else:
                    axes[i, j].imshow(transformed_image, cmap='gray')
                j = j + 1
            i = i + 1

        if self.save_flag:
            plt.savefig("transformed_multy_grid", type="jpeg")
        plt.show()


def id_transform(img, landmarks):
    img_clone = img.copy()
    for i in [1, 5, 9, 13, 17]:
        for j in range(3):
            cv2.line(img_clone,
                     (int(landmarks[i + j][0].item()), int(landmarks[i + j][1].item())),
                     (int(landmarks[i + j + 1][0].item()), int(landmarks[i + j + 1][1].item())), (255, 0, 0), 2)
    return img_clone
