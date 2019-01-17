from __future__ import division, print_function, absolute_import

import os
from io import StringIO
# import warnings
import sys
import argparse
from pathlib import Path

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


# warnings.filterwarnings('ignore')

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

# 0 в вагоне, 3 мб выход.
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

def print_finals(enter_train, exit_train):
    NO_PEOPLE = True
    for el in enter_train:
        print(el[0])
        NO_PEOPLE = False
    for el in exit_train:
        print(el[0])
        NO_PEOPLE = False
    return NO_PEOPLE

#     =============================================================================================
#     =============================================================================================
# Files changed:
# demo.py, deep_sort/tracker.py, deep_sort/track.py
#     =============================================================================================

def main(
    video_path: Path,
    out_dir: Path,
    yolo: YOLO,
    metric,
    tracker,
    encoder,
):
    nms_max_overlap = 1.0
    writeVideo_flag = True

    video_capture = cv2.VideoCapture(video_path.as_posix())
    
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out_name = 'DeepSort_' + video_path.name
        out = cv2.VideoWriter((out_dir/out_name).as_posix(), fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    num_frame = 0
    exit = [500, 100, 1400, 750]
    active_tracks_exists = False
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        num_frame += 1
        if num_frame % 12 != 0 and (not active_tracks_exists):
            continue

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)
        features = encoder(frame, boxs)

        # score to 1.0 here.
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

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
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
        
    enter_train = []
    exit_train = []
    
    enter_train, exit_train = postproc(tracker.old_conf_tracks, False, enter_train, exit_train)
    enter_train, exit_train = postproc(tracker.tracks, True, enter_train, exit_train)
    
    NO_PEOPLE = print_finals(enter_train, exit_train)
        
    with open(out_dir/'bus_report_all.csv', 'a+') as f:
        if NO_PEOPLE == True: 
            #s = StringIO(video_path.name + ',' + 'empty' + ',' + '-' + ',' + '-' + '\n') # name,empty,-,-\n
            f.write(video_path.name + ',' + 'empty' + ',' + '-' + ',' + '-' + '\n')        
        else:
            #s = StringIO(video_path.name + ',' + '-----' + ',' + str(len(enter_train)) + ',' + str(len(exit_train)) + '\n')
            f.write(video_path.name + ',' + '-----' + ',' + str(len(enter_train)) + ',' + str(len(exit_train)) + '\n')
    
    with open(out_dir/'detailed_report{}.txt'.format(video_path.name), 'w+') as f:
        if NO_PEOPLE == True:
            f.write("%s " % ('==EMPTY=='))
        else:
            f.write("%s " % ('IN (id): '))
            f.write("%s \n" % (" ".join(str(x[0]) for x in enter_train)))
            f.write("%s " % ('OUT (id): '))
            f.write("%s \n" % (" ".join(str(x[0]) for x in exit_train)))

        if len(tracker.old_conf_tracks) != 0 or len(tracker.tracks) !=0:
            f.write("\n%s \n" % ('==MOVEMENT INSIDE BUS=='))

        for track in tracker.old_conf_tracks:
            if track.last_seen_frame == -1:
                track.last_seen_frame = num_frame
            if track.last_seen_frame - track.start_frame > 25:
                f.write("%s \n" % ('track ' + str(track.track_id) + ' :'+str(track.start_frame) +'--'+ str(track.last_seen_frame) ))
                # берем best-practice codestyle, и выкидываем его
                # бывает(
        for track in tracker.tracks:
            if track.last_seen_frame == -1:
                track.last_seen_frame = num_frame
            if track.last_seen_frame - track.start_frame > 25:
                f.write("%s \n" % ('track ' + str(track.track_id) + ' :'+str(track.start_frame) +'--'+ str(track.last_seen_frame) ))


if __name__ == '__main__':
    # I guess code below doesn't work now
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
        main(cur_video, args, yolo, metric, tracker, encoder)
        tracker.reset()
