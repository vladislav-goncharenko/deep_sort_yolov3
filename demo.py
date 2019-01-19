import os
from io import StringIO
import sys
import argparse
from pathlib import Path
import json
from contextlib import ExitStack

import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from utils import open_video, frames


out_file_prefix = 'ds_' # stands for DeepSort

def center_x(bb):
    return (bb[2] + bb[0]) / 2

def center_y(bb):
    return (bb[3] + bb[1]) / 2

def center(bb):
    return np.asarray([center_x(bb), center_y(bb)])

def belong(bb1, bb2, eps=20):
    c_x = center_x(bb2)
    c_y = center_y(bb2)
    if (c_x < bb1[2] and c_x > bb1[0] and c_y < bb1[3] and c_y > bb1[1]):
        return True
    return False

def print_final_state(tracks, input_video):
    for track in tracks:
        print('last detection on frame: ', track.num_frame_end, '\n')
        print("Трек №", track.track_id, " завершился в состоянии ", track.walk_history[-1])

def postproc(tracks, is_new, enter_train, exit_train):
    for track in tracks:
        
        if (is_new):
            if not track.is_confirmed():
                    continue
        else:
            if  track.last_seen_frame - track.start_frame < 25:
                continue

        # USUAL
        if track.track_id == 6:
            print(is_new)
            print(len(track.walk_history), track.start_frame, track.last_seen_frame)
            for el in track.walk_history:
                print(el)
            print('\n')

        if track.begin_position == 3 and track.walk_history[-1] == 0:
            enter_train.append([track.track_id, track.change_env_frame])
            continue

        if track.begin_position == 0 and track.walk_history[-1] == 3:
            exit_train.append([track.track_id, track.change_env_frame])
            continue        
    return enter_train, exit_train

def process_video(
    video_path: Path,
    out_dir: Path,
    yolo: YOLO,
    metric,
    tracker: Tracker,
    encoder,
    *,
    write_video=False,
    write_yolo=False,
) -> tuple:
    '''
    Processes one video with given objects

    returns ('filename', in_count', 'out_count')
    '''
    # parameters of the algorithm, hardcoded for now
    nms_max_overlap = 1.0
    exit = [500, 100, 1400, 750]
    yolo_boxes = []

    with ExitStack() as stack:
        in_video = stack.enter_context(open_video(video_path))
        out_path = out_dir/(out_file_prefix + video_path.name)
        if write_video:
            width = in_video.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            out_video = stack.enter_context(open_video(
                out_path,
                'w',
                cv2.VideoWriter_fourcc(*'XVID'), # fourcc
                15, # fps
                (width, height), # frame size
            ))

        active_tracks_exists = False

        for num_frame, frame in enumerate(frames(in_video)):
            if num_frame % 12 != 0 and (not active_tracks_exists):
                continue

            image = Image.fromarray(frame)
            boxs = yolo.detect_image(image)
            yolo_boxes.append(boxs)
            features = encoder(frame, boxs)

            # score to 1.0 here.
            detections = [Detection(bbox, 1.0, feature)
                            for bbox, feature in zip(boxs, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            active_tracks_exists = False

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr() 
                if belong(exit, bbox, eps=20):
                    if track.begin_position == -1:
                        track.begin_position = 3
                        track.num_frame_begin = num_frame
                    else:
                        track.change_env_frame = num_frame
                    track.walk_history = np.append(track.walk_history, 3)
                    continue

                # если появился новый трек внутри маршрутки + in_train
                if track.begin_position == -1:
                    track.begin_position = 0
                    track.num_frame_begin = num_frame
                else:
                    track.change_env_frame = num_frame
                track.walk_history = np.append(track.walk_history, 0)

            for track in tracker.tracks:
                bbox = track.to_tlbr()                
                if not track.is_confirmed() or track.time_since_update > 1: 
                    continue

                if track.start_frame == -1:
                    track.start_frame = num_frame
                track.last_seen_frame = num_frame

                if not belong(exit, bbox, eps=20): #просто не будет отрисовки + если никого нет в зоне выхода, то 
                    continue                        # шаг в 12 кадров

                active_tracks_exists = True
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
                cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
                # 1080 1920
                # cv2.rectangle(frame, (500, 100), (1400, 1000),(0,255,0), 3)

            if write_video:
                out_video.write(frame)
    
    if write_yolo:
        with open(out_path.with_suffix('.yolo.json')) as yolo_file:
            json.dump(yolo_boxes, yolo_file)

    enter_train = []
    exit_train = []
    enter_train, exit_train = postproc(tracker.old_conf_tracks, False, enter_train, exit_train)
    enter_train, exit_train = postproc(tracker.tracks, True, enter_train, exit_train)
    movement = {}
    for tracks in [tracker.old_conf_tracks, tracker.tracks]:
        for track in tracks:
            if track.last_seen_frame == -1:
                track.last_seen_frame = num_frame
            if track.last_seen_frame - track.start_frame > 25:
                movement[track.track_id] = {
                    'start': track.start_frame,
                    'end': track.last_seen_frame,
                }
    out_json = {
        'in_ids': enter_train,
        'out_ids': exit_train,
        'movement': movement,
    }
    with open(out_path.with_suffix('.json'), 'w') as json_file:
        json.dump(out_json, json_file)

    return video_path.name, len(enter_train), len(exit_train)


if __name__ == '__main__':
    # code below doesn't work now
    # see perform_demo.ipynb

    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--input_video", required=True, 
    #                 help="path to input video")
    ap.add_argument("-i", "--input_dir", required=True, 
                    help="path to input videos folder, like /mnt/nfs/buses/bu/3/")
    ap.add_argument("-r", "--report_dir", required=True, 
                    help="path to input videos folder, like /mnt/nfs/.../")
    ap.add_argument("-o", "--output_video_dir", required=True, 
                    help="path to the output folder")
    args = vars(ap.parse_args())

    directory = args['input_dir']
    files = os.listdir(directory) 

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None

   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    yolo = YOLO()
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    for cur_video in files:
        if '.filepart' in cur_video:
            print('skip ', cur_video, ', it is broken')
            continue

        print('\n\nvideo = ', cur_video)
        process_video(cur_video, args, yolo, metric, tracker, encoder)
        tracker.reset()
