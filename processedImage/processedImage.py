import numpy as np
import torch
import cv2


class ProcessedImage(object):
#     CR: doc? why do you have a class without functionality?
    def __init__(self, row):
        self.row = int(row)

        self._images_path = r"D:\TAU\Research\Project\Dataset\GANeratedDataset_v3\GANeratedHands_Release\data\noObject\\"
        self._csv_path = r"D:\TAU\Research\Project\Learning\Results\LandmarksExtraction\raw_dataframes_clean_fromt_hand.csv"

        # load and extract data
        with open(self._csv_path, 'r') as csv:
            line = csv.readlines()[self.row]
            line_values = line.split(',')
        image_name = line_values[0].split('_', 1)
        # CR: use pathlib - this way it will work no matter what os u are running on
        full_image_path = self._images_path + image_name[0] + '\\' + image_name[1]
        landmarks = np.array(line_values[1:], dtype=float)

        self.landmarks = torch.from_numpy(landmarks).view(-1, 2)
        self.img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
