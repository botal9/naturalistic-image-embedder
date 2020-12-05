from http import HTTPStatus
import os.path as path
import requests
from typing import Dict, Any

from hdr_client.html_parser import HdrHtmlParser
from common.error import ImageEmbedderError
from common.lighting import LightingType


class HdrClientError(ImageEmbedderError):
    def __init__(self, message):
        super().__init__(message)


class HdrClient:
    def __init__(self):
        self.parser = HdrHtmlParser()
        self.outdoor_demo_url = 'http://rachmaninoff.gel.ulaval.ca:8000/'
        self.indoor_demo_url = 'http://rachmaninoff.gel.ulaval.ca:8001/'

    def get_hdr_generator_url(self, lighting_type: LightingType):
        if lighting_type == LightingType.INDOOR:
            return self.indoor_demo_url
        elif lighting_type == LightingType.OUTDOOR:
            return self.outdoor_demo_url
        else:
            raise HdrClientError(f'Undefined lighting type: {lighting_type}')

    def get_hdr_image(self, ldr_image_path: str, lighting_type: LightingType) -> (bytes, Dict[str, Any]):
        """Takes single LDR image path and required lighting type. Returns bytes of HDR environment map in ".exr" format
        and additional information about given image, if such exists.
        """
        if not path.isfile(ldr_image_path):
            raise FileNotFoundError(ldr_image_path)

        demo_url = self.get_hdr_generator_url(lighting_type)
        demo_response = requests.post(demo_url, files={'file': (ldr_image_path, open(ldr_image_path, 'rb'))})
        if demo_response.status_code != HTTPStatus.OK:
            raise HdrClientError(f'Got {demo_response.status_code} on {demo_url} request')

        self.parser.parse(demo_response.text)
        image_download_path = None
        for link in self.parser.links:
            if 'download' in link:
                image_download_path = link
        if image_download_path is None:
            self.parser.reset()
            raise HdrClientError(f'Image download link not found for image {ldr_image_path}')

        additional_data = {}
        for i in range(2, len(self.parser.additional_info), 2):
            additional_data[self.parser.additional_info[i]] = float(self.parser.additional_info[i + 1])
        self.parser.reset()

        image_url = demo_url + image_download_path[2:]  # remove "./"
        image_response = requests.get(image_url)
        if image_response.status_code != HTTPStatus.OK:
            raise HdrClientError(f'Got {image_response.status_code} on {image_url} request')

        return image_response.content, additional_data

    def download_hdr_image(self, ldr_image_path: str,
                           lighting_type: LightingType, hdr_image_path: str) -> Dict[str, Any]:
        image_data, additional_data = self.get_hdr_image(ldr_image_path, lighting_type)
        with open(hdr_image_path, 'wb') as image:
            image.write(image_data)
        return additional_data
