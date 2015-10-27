import cv2
import numpy as np


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


def crop_sample_image(src_image, region):
    left, top, right, bottom = polygon2rectangle(region)

    pattern = np.zeros(src_image.shape, src_image.dtype)
    cv2.fillPoly(pattern, [np.array(region, 'int32')], 255)

    pattern = pattern[top:bottom, left:right]
    rect_image_segment = src_image[top:bottom, left:right]

    #result_image = rect_image_segment.copy()
    #print(rect_image_segment.shape, result_image.shape, pattern.shape)
    #cv2.seamlessClone(rect_image_segment, result_image, pattern, (0, 0), cv2.NORMAL_CLONE)
    #return result_image

    for yi in range(rect_image_segment.shape[0]):
         for xi in range(rect_image_segment.shape[1]):
             if pattern[yi][xi] == 0:
                 rect_image_segment[yi][xi] = 0 if len(rect_image_segment.shape) < 3 else (0, 0, 0)

    return rect_image_segment