#!/usr/bin/env python

import logging
import cv2
import numpy
import argparse
import pickle


class FragmentView(object):
    def __init__(self, img, name):
        self.img = img
        cv2.imshow(name, img)


class ROICtrl(object):
    COLOR_ROI_MARGIN = (0, 255, 0)
    COLOR_ROI_POINT = (255, 0, 0)

    def __init__(self, img, ROI, window_name):
        self._points = []
        self.img = img
        self.name = window_name
        self.selected_region = None
        self.ROI = ROI if ROI is not None else []

        cv2.imshow(self.name, self.img)
        cv2.setMouseCallback(window_name, self.__mouse_event_callback)
        self.__update_points()

    def __hightlight_ROI(self, img, points):
        for point in points:
                cv2.circle(img, point, 3, self.COLOR_ROI_POINT, 3)

        if len(points) > 3:
            cv2.line(img, points[0], points[-1], self.COLOR_ROI_MARGIN, 2)

            for i in range(1, len(points)):
                cv2.line(img, points[i-1], points[i], self.COLOR_ROI_MARGIN, 2)

    def __update_points(self):
        if len(self._points) > 0 or len(self.ROI) > 0:
            img = self.img.copy()
            self.__hightlight_ROI(img, self._points)
            for points in self.ROI:
                self.__hightlight_ROI(img, points)

            cv2.imshow(self.name, img)

    def __mouse_event_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self._points) < 4:
                self._points.append((x, y))

            if (len(self._points) == 4):
                self.ROI.append(self._points)
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
        img = self.roi_ctrl_view.img
        for region in self.roi_ctrl_view.ROI:
            pass

        gray_mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges_mat = gray_mat.copy()
        edges_mat = cv2.Canny(edges_mat, 100, 200)
        cv2.imshow("edges", edges_mat)

    def save(self):
        assert self.project_path is not None
        cached_data = {'ROI': self.roi_ctrl_view.ROI}
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