#!/usr/bin/env python
import socket
import sys
from http.client import HTTPConnection
import logging

logger = logging.getLogger(__file__)

class ImageFeed(object):
    def __init__():
        pass


class WebCamera(ImageFeed):
    def __init__(self, url, request):
        self.url = url
        self.request = request

    def get_image(self):
        conn = HTTPConnection(self.url)
        conn.request("GET", self.request)
        fh = conn.sock.makefile(mode="rb")

        # Read in HTTP headers:
        line = fh.readline().decode()

        while line.strip() != '':
            parts = line.split(':')
            if len(parts) > 1 and parts[0].lower() == 'content-type':
                # Extract boundary string from content-type
                content_type = parts[1].strip()
                boundary = content_type.split(';')[1].split('=')[1]
            line = fh.readline().decode()

        if not boundary:
            raise Exception("Can't find content-type")
        logger.debug("boundary=%s" % boundary)
        # Seek ahead to the first chunk
        while line.strip() != "--%s" % boundary:
            line = fh.readline().decode()

        # Read in chunk headers
        while line.strip() != '':
            parts = line.split(':')
            if len(parts) > 1 and parts[0].lower() == 'content-length':
                # Grab chunk length
                length = int(parts[1].strip())
            line = fh.readline().decode()

        image = fh.read(length)
        logger.debug("image has been received - size: %d" % len(image))
        return image

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    raw_image = WebCamera("192.168.1.187:8080", "/?action=stream").get_image()
