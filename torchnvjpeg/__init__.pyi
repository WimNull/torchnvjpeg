#
# Automatically generated file, do not edit!
#

from __future__ import annotations
import lib
import torch
import typing

__all__ = [
    "NvJpeg"
]


class NvJpeg():
    def __init__(self, device_id: int = 0, stream: object = None, max_image_size: int = 24883200, 
        batch_size: int = 2, max_cpu_threads: int = 1, device_padding: int = 0, host_padding: int = 0, gpu_huffman: bool = True) -> None: 
        """
        Initialize nvjpeg batch decoder.
        Parameters:
            device_id: int
            max_cpu_threads: int
            stream: torch.cuda.Stream
            max_image_size: int
            device_padding: int
            host_padding: int
            gpu_huffman: bool
        """
    def batch_decode(self, data: typing.List[str], stream_sync: bool = True) -> typing.List[torch.Tensor]: 
        """
        Decode list of images to list of torch cuda tensor.
        Parameters:
            data: List[string], list of image bytes
            stream_sync: bool, whether to do steam.synchronize()
        Returns:
            list of image cuda tensor in HWC foramt.
        """
    @staticmethod
    def decode(data: str, stream_sync: bool = True) -> typing.Any: 
        """
        Decode image to torch cuda tensor.
        Parameters:
            data: string, image bytes
            stream_sync: bool, whether to do steam.synchronize()
        Returns:
            image cuda tensor in HWC foramt.
        """
    @staticmethod
    def encode(img: torch.cuda.Tensor, quality: int=75, format: str='RGB') -> typing.Any: 
        """
        encode torch cuda image to bytes.
        Parameters:
            img: torch.cuda.Tensor, image 
            quality: int, encode quality 
            format: string, 'RGB' or 'BGR'
        Returns:
            string, image bytes.
        """
    def get_device_id(self) -> int: ...
    pass
