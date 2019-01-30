from contextlib import contextmanager
from pathlib import Path

import cv2


@contextmanager
def open_video(video_path: Path, mode: str='r', *args):
    '''
    Context manager to work with cv2 videos

    Modes are either 'r' for read or 'w' write
    which returns cv2.VideoCapture or cv2.VideoWriter respectively.
    Additional arguments passed according to OpenCV documentation
    '''
    if video_path is None:
        yield None
    else:
        if mode == 'r':
            video = cv2.VideoCapture(video_path.as_posix(), *args)
        elif mode == 'w':
            video = cv2.VideoWriter(video_path.as_posix(), *args)
        else:
            raise ValueError('Incorrect open mode "{}", "r" or "w" expected!'.format(mode))
        if not video.isOpened(): raise ValueError('Video {} is not opened!'.format(video_path))
        try:
            yield video
        finally:
            video.release()

def frames(video: cv2.VideoCapture):
    '''
    Generator of frames of the video provided
    '''
    while True:
        retval, frame = video.read()
        if not retval:
            break
        # this is to return RGB image instead of BGR
        frame = frame[..., ::-1]
        yield frame

def convert_bbox(bbox: tuple, fr: str, to: str) -> tuple:
    '''
    Converts bounding box from one fromat to other

    Available formats:
        * 'xywh' - top left point and width, height
        * 'tlbr' - top left point (x, y) bottom right point (x, y)

    Note: make enum for `fr`, `to`
    '''
    if fr == 'xywh' and to == 'tlbr':
        x, y, w, h = bbox
        return [ x, y, x + w, y + h ]
    elif fr == 'tlbr' and to == 'xywh':
        l, t, r, b = bbox
        return [l, t, r - l, b - t]

    raise NotImplementedError('sorry, this functionality is not currently available')

def prettify_bbox(image, bbox: list) -> list:
    '''
    Validates and shrinks bounding box in format 'xywh'
    to be succesfully painted by cv2
    '''
    y_size, x_size = image.shape[:2]
    l, t, r, b = convert_bbox(bbox, 'xywh', 'tlbr')
    if l > x_size:
        raise ValueError('left corner is righter than image size: {} vs {}'.format(l, x_size))
    if t > y_size:
        raise ValueError('top is downer than image size: {} vs {}'.format(t, y_size))
    return convert_bbox(
        [max(l, 0), max(t, 0), min(r, x_size), min(b, y_size)],
        'tlbr',
        'xywh',
    )
