import cv2
import numpy as np
import logging
import tempfile
import os


def polygon2rectangle(region):
    left = region[0][0]
    top = region[0][1]
    right = region[0][0]
    bottom = region[0][1]

    for point in region:
        left = min(left, point[0])
        right = max(right, point[0])
        top = min(top, point[1])
        bottom = max(bottom, point[1])

    return left, top, right, bottom


def compute_training_sample_shape(images):
    shape = []

    for i in range(max([len(image.shape) for image in images])):
        shape.append(max([image.shape[i] for image in images]))

    return shape


def average_color(image):
    """
    supported only RGB src image
    """
    color_channels_count = image.shape[-1] if len(image.shape) > 2 else 1
    color = np.ndarray(color_channels_count, dtype=image.dtype)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            color += image[y][x]
            color /= 2

    return color


def crop_sample_image(src_image, region, background_color=None):
    left, top, right, bottom = polygon2rectangle(region)

    pattern = np.zeros(src_image.shape[:2], src_image.dtype)
    cv2.fillPoly(pattern, [np.array(region, np.int32)], 255)
    pattern = pattern[top:bottom, left:right]
    rect_image_segment = src_image[top:bottom, left:right]

    # logging.debug('crop_sample_image shapes: %s - %s' % (rect_image_segment.shape, src_image.shape))
    # result_image = rect_image_segment.copy()
    # print(rect_image_segment.shape, result_image.shape, pattern.shape)
    # cv2.seamlessClone(rect_image_segment, result_image, pattern, (0, 0), cv2.NORMAL_CLONE)
    # return result_image

    _background_color = average_color(rect_image_segment) if background_color is None else background_color
    #
    # for yi in range(rect_image_segment.shape[0]):
    #     for xi in range(rect_image_segment.shape[1]):
    #         if pattern[yi][xi] == 0:
    #             rect_image_segment[yi][xi] = _background_color

    return rect_image_segment, pattern, _background_color


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    bin_n = 16
    # quantizing binvalues in (0...bin_n)
    bins = np.int32(bin_n * ang / (2 * np.pi))

    # Divide to 4 sub-squares
    height = img.shape[0]
    widht = img.shape[1]

    dh = int(height / 2)
    dw = int(widht / 2)

    bin_cells = bins[:dh, :dw], bins[dh:, :dw], bins[:dh, dw:], bins[dh:, dw:]
    mag_cells = mag[:dh, :dw], mag[dh:, :dw], mag[:dh, dw:], mag[dh:, dw:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist


class StatModel(object):
    def load(self, fn):
        logging.info(dir(self.model))
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

    def serialize(self):
        data_path = tempfile.mktemp()
        self.save(data_path)

        with open(data_path, 'rb') as fd:
            data = fd.read()

        os.unlink(data_path)
        return data

    def deserialize(self, data):
        data_path = tempfile.mktemp()
        with open(data_path, 'wb') as fd:
            fd.write(data)

        self.load(data_path)
        os.unlink(data_path)


class KNearest(StatModel):
    def __init__(self, k=3):
        self.k = k
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()


class SVM(StatModel):
    def __init__(self, C=0.5, gamma=1.0):
        self.params = dict(kernel_type=cv2.SVM_RBF,
                           svm_type=cv2.SVM_C_SVC,
                           C=C,
                           gamma=gamma)
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        for sample in samples:
            logging.debug('svm: training sample shape - %s' % sample.shape)
        self.model.train(samples, responses, params=self.params)

    def predict(self, samples):
        for i in range(len(samples)):
            logging.debug('svm: predict(samples[%d].shape=%s)' % (i, samples[i].shape))
        return self.model.predict_all(samples).ravel()
