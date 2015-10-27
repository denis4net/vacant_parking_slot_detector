#!/usr/bin/env python

import logging
import cv2
import numpy as np
import argparse
import pickle
import algoritms

class FragmentView(object):
    def __init__(self, img, title):
        self.img = img
        self.title = title

    def show(self):
        cv2.imshow(self.title, self.img)


class MultipleImagesView(object):
    def __init__(self, title, images):
        self.images = images
        self.title = title
        self.max_img_width = max([image.shape[1] for image in self.images])
        self.max_img_height = max([image.shape[0] for image in self.images])

        shape = (self.max_img_height, (self.max_img_width + 2) * len(images)) if len(images[0].shape) < 3 else  (self.max_img_height, (self.max_img_width + 2) * len(images), images[0].shape[2])
        self.image = np.zeros(shape, dtype=images[0].dtype)
        current_x = 0
        current_y = 0

        for image in self.images:
            logging.debug("%s - %s" % (self.image.shape, image.shape))
            self.image[current_y:current_y+image.shape[0], current_x:current_x+image.shape[1]] = image
            current_x += image.shape[1] + 2

    def show(self):
        cv2.imshow(self.title, self.image)


class ROI(object):
    class Type:
        FREE = 1
        BUSY = 2

    def __init__(self, points, type):
        self.points = points
        self.type = type

    def is_free(self):
        return self.type == ROI.Type.FREE

    def is_busy(self):
        return self.type == ROI.Type.BUSY


class ROICtrl(object):
    COLOR_FREE_ROI_MARGIN = (0, 255, 0)
    COLOR_BUSY_ROI_MARGIN = (0, 0, 255)
    COLOR_ROI_POINT = (255, 0, 0)

    def __init__(self, img, ROIs, window_name):
        self._points = []
        self.img = img
        self.name = window_name
        self.selected_region = None
        self.ROIs = ROIs if ROIs is not None else []
        self.roi_type = ROI.Type.FREE

        cv2.imshow(self.name, self.img)
        cv2.setMouseCallback(window_name, self.__mouse_event_callback)
        self.__update_points()

    def __hightlight_region(self, img, points, color):
        for point in points:
                cv2.circle(img, point, 3, self.COLOR_ROI_POINT, 3)

        if len(points) > 3:
            cv2.line(img, points[0], points[-1], color, 2)

            for i in range(1, len(points)):
                cv2.line(img, points[i-1], points[i], color, 2)

    def __update_points(self):
        if len(self._points) > 0 or len(self.ROIs) > 0:
            img = self.img.copy()
            self.__hightlight_region(img, self._points, self.COLOR_FREE_ROI_MARGIN)

            for roi in self.ROIs:
                self.__hightlight_region(img, roi.points, self.COLOR_FREE_ROI_MARGIN if roi.is_free() else self.COLOR_BUSY_ROI_MARGIN)

            cv2.imshow(self.name, img)

    def __mouse_event_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            if event == cv2.EVENT_RBUTTONDOWN:
                self.roi_type = ROI.Type.FREE
            else:
                self.roi_type = ROI.Type.BUSY

            if len(self._points) < 4:
                self._points.append((x, y))

            if len(self._points) == 4:
                self.ROIs.append(ROI(self._points, self.roi_type))
                self._points = []

            self.__update_points()

        elif event == cv2.EVENT_MOUSEMOVE:
            pass


class Application(object):

    def __init__(self):
        self.roi_ctrl_view = None
        self.project_path = None

    def init(self, src_img_path, project_path):
        self.project_path = project_path
        src_img = None if src_img_path is None else cv2.imread(src_img_path)

        ROI = []
        try:
            cached_data = pickle.load(open(project_path, "rb"))
            ROI = cached_data['ROI']
        except (ValueError, OSError, IOError, pickle.UnpicklingError) as e:
            logging.error('ROI settings was not loaded: %s' % e)

        self.roi_ctrl_view = ROICtrl(src_img, ROI, "Source image view")

    def process(self):
        # denoise: cv2.fastNlMeansDenoisingColored()
        img = self.roi_ctrl_view.img
        # cv2.imshow('denoised', img)
        gray_mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges_mat = gray_mat.copy()
        edges_mat = cv2.Canny(edges_mat, 100, 200)

        roi_images = []
        for roi in self.roi_ctrl_view.ROIs:
            sample = algoritms.crop_sample_image(edges_mat, roi.points)
            roi_images.append(sample)

        if len(roi_images) > 0:
            logging.debug("training image shape: %s", algoritms.compute_training_sample_shape(roi_images))
            miv = MultipleImagesView("ROI", roi_images)
            miv.show()

    def save(self):
        assert self.project_path is not None
        cached_data = {'ROI': self.roi_ctrl_view.ROIs}
        pickle.dump(cached_data, open(self.project_path, "wb"))

    def event_loop(self):
        assert self.roi_ctrl_view is not None
        cv2.waitKey(0)
        self.save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='project file path')
    parser.add_argument('--src-image', help='source image file')
    args = parser.parse_args()

    app = Application()
    app.init(args.src_image, args.project)
    app.process()
    app.event_loop()