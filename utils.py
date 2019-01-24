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
            video = cv2.VideoCapture(video_path.as_posix())
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
