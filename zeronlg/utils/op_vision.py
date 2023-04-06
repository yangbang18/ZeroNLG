import requests
import numpy as np
from PIL import Image
from typing import List, Tuple, Union


def get_uniform_frame_ids(
        num_total_frames: int, 
        num_frames: int,
    ) -> List[int]:
    
    if num_total_frames <= num_frames:
        frame_ids = [_ for _ in range(num_total_frames)]
        frame_ids = frame_ids + [frame_ids[-1]] * (num_frames - num_total_frames)
        return frame_ids

    # there will be num_frames intervals
    ids = np.linspace(0, num_total_frames, num_frames + 1)
    frame_ids = []
    for i in range(num_frames):
        # get the middle frame index of each interval
        frame_ids.append(round((ids[i] + ids[i+1]) / 2))
    return frame_ids


def process_images(
        images: Union[str, List[str], Image.Image, List[Image.Image], List[List[Image.Image]]], 
        num_frames: int = 8
    ) -> Tuple[List[Image.Image], bool, int, int]:

    images = [images] if not isinstance(images, list) else images
    batch_size = len(images)

    num_images_per_input, is_video = None, False
    if type(images[0]) is str:
        if images[0].startswith("http://") or images[0].startswith("https://"):
            # load images from remote URLs
            images = [Image.open(requests.get(item, stream=True).raw) for item in images]
        elif images[0].endswith('.mp4') or images[0].endswith('.avi'):
            # load local videos
            import decord
            is_video = True
            frames = []
            for item in images:
                reader = decord.VideoReader(item)
                this_frames = reader.get_batch(get_uniform_frame_ids(len(reader), num_frames)).asnumpy()
                this_frames = [Image.fromarray(frame) for frame in this_frames]
                frames.extend(this_frames)
            images = frames
        else:
            # load local images
            images = [Image.open(item) for item in images]
    elif isinstance(images[0], list):
        assert isinstance(images[0][0], Image.Image), type(images[0][0])
        num_images_per_input = len(images[0])
        is_video = num_images_per_input > 1
        images = [images[i][j] for i in range(len(images)) for j in range(num_images_per_input)]
    else:
        assert isinstance(images[0], Image.Image), type(images[0])
    
    return images, is_video, num_images_per_input or num_frames, batch_size
