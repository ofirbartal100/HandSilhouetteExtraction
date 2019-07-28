import numpy as np
import torch
import cv2
import abc


class ProcessedImageTransform(abc.ABC):
    def __init__(self, show_landmarks_flag):
        self._show_landmarks_flag = show_landmarks_flag

    @abc.abstractmethod
    def _transform(self, processedImage):
        pass

    def transform(self, processedImage):
        transformed_img = self._transform(processedImage)
        if self._show_landmarks_flag:
            return self._show_landmarks(transformed_img, processedImage.landmarks)
        return transformed_img

    def _show_landmarks(self, img, landmarks):
        img_clone = img.copy()
        # CR: you used this array more than one - put it in a const. also, isn't that functionality a code duplication?
        for i in [1, 5, 9, 13, 17]:
            for j in range(3):
                cv2.line(img_clone,
                         (int(landmarks[i + j][0].item()), int(landmarks[i + j][1].item())),
                         (int(landmarks[i + j + 1][0].item()), int(landmarks[i + j + 1][1].item())), (255, 0, 0), 2)
        return img_clone


class CannyTransform(ProcessedImageTransform):
    def __init__(self, l_hesteresys, h_hesteresys, show_landmarks_flag=False):
        super().__init__(show_landmarks_flag)
        self.low_hesteresys = l_hesteresys
        self.high_hesteresys = h_hesteresys

    def _transform(self, processedImage):
        return cv2.Canny(processedImage.img, self.low_hesteresys, self.high_hesteresys)


class IdentityTransform(ProcessedImageTransform):
    def __init__(self, show_landmarks_flag=False):
        super().__init__(show_landmarks_flag)

    def _transform(self, processedImage):
        return processedImage.img.copy()


class SteeredEdgeTransform(ProcessedImageTransform):
    def __init__(self, k_size, mask_size, show_landmarks_flag=False):
        super().__init__(show_landmarks_flag)
        self._mask_size = mask_size
        self._k_size = k_size

    def _transform(self, processedImage):
        transformed_img = np.zeros(processedImage.img.shape[:2], np.uint8)
        for i in [1, 5, 9, 13, 17]:
            for j in range(3):
                hist_mask = Mask(processedImage.img.shape[:2], processedImage.landmarks[i + j], self._mask_size)

                kernel_down, kernel_up = self.create_kernel(processedImage.landmarks[i + j],
                                                            processedImage.landmarks[i + j + 1], self._k_size)

                filtered_down = cv2.filter2D(processedImage.img, cv2.CV_8U, kernel_down)
                filtered_up = cv2.filter2D(processedImage.img, cv2.CV_8U, kernel_up)

                filtered_down = hist_mask(filtered_down)
                filtered_up = hist_mask(filtered_up)

                transformed_img = cv2.bitwise_or(transformed_img, filtered_down)
                transformed_img = cv2.bitwise_or(transformed_img, filtered_up)

        return transformed_img

    def create_kernel(self, source, target, kernel_size):
        direction = target - source
        # flip since image is in flipped coordinates
        direction *= torch.from_numpy(np.array([1.0, -1.0]))
        direction *= 1 / direction.norm() #CR: why not just /= direction.norm()?
        #CR: what is 5?, put the array in a const that will tell me something about it. I don't understand where those numbers come from.
        if kernel_size == 5:
            sobely = torch.from_numpy(np.array([
                [2., 1., 0., -1., -2.],
                [2., 1., 0., -1., -2.],
                [4., 2., 0., -2., -4.],
                [2., 1., 0., -1., -2.],
                [2., 1., 0., -1., -2.],
            ]))
        else:
            sobely = torch.from_numpy(np.array([
                [1., 0., -1.],
                [2., 0., -2.],
                [1., 0., -1.]
            ]))
        sobelx = sobely.t()

        direction_filter = sobelx * direction[0] + sobely * direction[1]
        return direction_filter.numpy(), direction_filter.numpy() * -1


class Mask(object):
    def __init__(self, shape, point, mask_size):
        self._mask = np.zeros(shape, np.uint8)
        x_range_start, x_range_stop, y_range_start, y_range_stop = Mask.extract_mask_location(point, mask_size)
        self._mask[y_range_start:y_range_stop, x_range_start:x_range_stop] = 255

    @staticmethod
    def extract_mask_location(point, mask_size):
        return int(point[0] - mask_size / 2), int(point[0] + mask_size / 2), int(point[1] - mask_size / 2), int(
            point[1] + mask_size / 2)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            return self._mask
        try:
            return cv2.bitwise_and(*args, *args, mask=self._mask)
        except:
            raise
