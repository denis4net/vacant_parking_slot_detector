#!/usr/bin/env python2

import logging
import cv2
import numpy as np
import argparse
import pickle
import algoritms
import copy


class EventEmiter(object):
    def __init__(self):
        self.events = {}

    def on(self, name, handler):
        if name not in self.events:
            self.events[name] = []

        self.events[name].append(handler)

    def emit(self, name, **kwargs):
        logging.debug('EventEmiter: emit(%s)' % name)
        if name in self.events:
            for handler in self.events[name]:
                handler(**kwargs)


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

        shape = (self.max_img_height, (self.max_img_width + 2) * len(images)) if len(images[0].shape) < 3 else (self.max_img_height, (self.max_img_width + 2) * len(images), images[0].shape[2])
        self.image = np.zeros(shape, dtype=images[0].dtype)
        current_x = 0
        current_y = 0

        max_x = 0
        max_y = 0

        for image in self.images:
            max_y = max(max_y, image.shape[0])
            max_x = max(max_x, image.shape[1])

        images = []
        for image in self.images:
            images.append(cv2.resize(image, (max_x, max_y), interpolation=cv2.INTER_CUBIC))

        for image in images:
            self.image[current_y:current_y + image.shape[0], current_x:current_x + image.shape[1]] = image
            current_x += image.shape[1] + 2

    def show(self):
        cv2.imshow(self.title, self.image)


class HOGDEscriptorsView(MultipleImagesView):
    def __init__(self, descriptors):
        assert len(descriptors) > 0
        visualised_descriptors = []

        import math
        normalizer = 0
        for descriptor in descriptors:
            normalizer = max(normalizer, descriptor.max())

        for descriptor in descriptors:
            d = descriptor.reshape(-1, math.sqrt(descriptor.shape[0])) / normalizer

            height = d.shape[0]
            width = d.shape[1]

            d = cv2.resize(d, (10 * width, 10 * height), interpolation=cv2.INTER_CUBIC)
            visualised_descriptors.append(d)

            logging.debug('descriptor shape: %s' % str(d.shape))

        logging.debug('descriptors view: %s' % visualised_descriptors)

        MultipleImagesView.__init__(self, 'descriptors visualization', visualised_descriptors)


class ROI(object):
    __counter = 0

    class Type:
        FREE = 1
        BUSY = 2

    def __init__(self, points, type):
        self.id = self.__counter
        self.__counter += 1

        self.points = points
        self.type = type

    def is_free(self):
        return self.type == ROI.Type.FREE

    def is_busy(self):
        return self.type == ROI.Type.BUSY


class ROICtrl(EventEmiter):
    """
    UI Windows that allow chose regions of interests
    """

    COLOR_FREE_ROI_MARGIN = (0, 255, 0)
    COLOR_BUSY_ROI_MARGIN = (0, 0, 255)
    COLOR_ROI_POINT = (255, 0, 0)

    MODE_TRAINING = 0
    MODE_PREDICTION = 1

    class Events:
        ROI_UPDATED = 'roi_updated'

    def __init__(self, img, ROIs, window_name):
        EventEmiter.__init__(self)

        self._points = []
        self.img = img
        self.name = window_name
        self.selected_region = None
        self.ROIs = ROIs if ROIs is not None else []
        self.predicted_ROIs = []

        self.roi_type = ROI.Type.FREE
        self.mode = self.MODE_TRAINING

        cv2.imshow(self.name, self.img)
        cv2.setMouseCallback(window_name, self.__mouse_event_callback)
        self.__update_points()

    def __hightlight_region(self, img, points, color):
        for point in points:
                cv2.circle(img, point, 3, self.COLOR_ROI_POINT, 3)

        if len(points) > 3:
            cv2.line(img, points[0], points[-1], color, 2)

            for i in range(1, len(points)):
                cv2.line(img, points[i - 1], points[i], color, 2)

    def redraw(self):
        self.__update_points()

    def __update_points(self):
        img = self.img.copy()

        if self.mode == self.MODE_TRAINING:
            if len(self._points) > 0 or len(self.ROIs) > 0:
                self.__hightlight_region(img, self._points, self.COLOR_FREE_ROI_MARGIN)

                for roi in self.ROIs:
                    self.__hightlight_region(img, roi.points, self.COLOR_FREE_ROI_MARGIN if roi.is_free() else self.COLOR_BUSY_ROI_MARGIN)

        elif self.mode == self.MODE_PREDICTION:
            for roi in self.predicted_ROIs:
                self.__hightlight_region(img, roi.points, self.COLOR_FREE_ROI_MARGIN if roi.is_free() else self.COLOR_BUSY_ROI_MARGIN)

        cv2.imshow(self.name, img)

    def __mouse_event_callback(self, event, x, y, flags, params):
        if self.mode == self.MODE_TRAINING:
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
                    self.emit(ROICtrl.Events.ROI_UPDATED, data=self.ROIs)

        self.__update_points()


class Application(object):
    IMAGE_VIEW_WINDOWS_NAME = "Image view"

    STATUS_BUSY = 1
    STATUS_FREE = 0

    def __init__(self):
        self.roi_ctrl_view = None
        self.project_path = None
        self.classificator = None

    def init(self, img_path, project_path):
        self.project_path = project_path
        src_img = None if img_path is None else cv2.imread(img_path)

        ROI = []
        try:
            cached_data = pickle.load(open(project_path, "rb"))
            ROI = cached_data['ROI']
            self._init_classificator(cached_data['classificator_data'])
        except (ValueError, OSError, IOError, pickle.UnpicklingError) as e:
            logging.error('settings was not loaded: %s' % e)

        self.roi_ctrl_view = ROICtrl(src_img, ROI, self.IMAGE_VIEW_WINDOWS_NAME)
        self.roi_ctrl_view.on(ROICtrl.Events.ROI_UPDATED, self.on_ROI_update)

    def on_ROI_update(self, **kwargs):
        busy = False
        free = False
        for roi in kwargs['data']:
            if roi.is_busy():
                busy = True
            else:
                free = True

        if free and busy:
            self.train()

    def _init_classificator(self, data=None):
        logging.info('creating classificator')
        self.classificator = algoritms.SVM()

        if data is not None:
            logging.debug("initializing classificator from provided data")
            self.classificator.deserialize(data)

    def build_hog_descriptors(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray_img, 100, 200)
        preprocessed_image = gray_img

        images = []
        descriptors = []
        responses = []
        patterns = []

        for roi in self.roi_ctrl_view.ROIs:
            sample, pattern, bg = algoritms.crop_sample_image(preprocessed_image, roi.points, background_color=None)
            patterns.append(pattern)

            hog = algoritms.hog(sample)
            descriptors.append(np.float32(hog))
            responses.append(self.STATUS_FREE if roi.is_free() else self.STATUS_BUSY)
            images.append(sample)

        desc_view = HOGDEscriptorsView(descriptors)
        desc_view.show()

        patterns_view = MultipleImagesView('patterns visualization', patterns)
        patterns_view.show()
        # logging.debug('hog responses: %s' % responses)

        return np.float32(descriptors), np.int32(responses), images, self.roi_ctrl_view.ROIs

    def is_trained(self):
        return self.classificator is not None

    def process(self, img):
        assert self.is_trained()
        descriptors, responses, images, ROIs = self.build_hog_descriptors(img)

        logging.debug('processing %d descriptors' % len(descriptors))
        predictions = self.classificator.predict(descriptors)
        logging.debug('predictions - %s' % predictions)

        predicted_ROIs = []
        assert len(predictions) == len(ROIs)

        for i in range(len(predictions)):
            logging.debug('hog - %s, prediction - %d' % (descriptors[i], predictions[i]))
            roi = copy.copy(ROIs[i])
            prediction = predictions[i]
            roi.type = ROI.Type.FREE if prediction == self.STATUS_FREE else ROI.Type.BUSY
            predicted_ROIs.append(roi)

        self.roi_ctrl_view.mode = ROICtrl.MODE_PREDICTION
        self.roi_ctrl_view.predicted_ROIs = predicted_ROIs
        self.roi_ctrl_view.redraw()

        return images, predicted_ROIs

    def train(self):
        training_data, responses, images, ROIs = self.build_hog_descriptors(self.roi_ctrl_view.img)

        if len(training_data) > 0:
            miv = MultipleImagesView("Regions of interests", images)
            miv.show()
            self._init_classificator()
            self.classificator.train(training_data, responses)

    def save(self):
        logging.info('saving training data to file \"%s\"' % self.project_path)

        assert self.project_path is not None
        classificator_data = self.classificator.serialize() if self.classificator is not None else None
        assert classificator_data is not None
        cached_data = {'ROI': self.roi_ctrl_view.ROIs,
                       'classificator_data': classificator_data
                       }

        pickle.dump(cached_data, open(self.project_path, "wb"))

    def event_loop(self):
        assert self.roi_ctrl_view is not None
        cv2.waitKey(0)
        self.save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='project file path')
    parser.add_argument('--image', help='source image file')
    args = parser.parse_args()

    app = Application()
    app.init(args.image, args.project)

    if app.is_trained() and args.image is not None:
        img = cv2.imread(args.image)
        images, rois = app.process(img)
        cutted = MultipleImagesView('Regions of interest visualization', images)
        cutted.show()

    app.event_loop()
