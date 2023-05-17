#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import time
from threading import Lock, Thread

import cv2
import numpy as np


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 2)
        # os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 2)
        # os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class WebcamStream(object):
    def __init__(self, src, run_async=False):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        print(f"Camera frame rate set to {self.stream.get(cv2.CAP_PROP_FPS)}.")

        (_, self.frame) = self.stream.read()

        self.name = src.split('/')[-1]

        self.stopped = False
        self.is_async = run_async
        self.new_frame = False
        self.lock = Lock()
        self.thread = None

    def start(self):
        # start the thread to read frames from the video stream
        if self.is_async:
            self.thread = Thread(target=self.update, name=self.name, args=())
            self.thread.daemon = True
            self.thread.start()
            return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            with suppress_stdout_stderr():
                (_, self.frame) = self.stream.read()
            with self.lock:
                self.new_frame = True

    def read(self):
        # To keep the desired frame rate, we must return the frame immediately, 
        # and not wait for the next frame to be read. However, the frame is
        # returned as a reference, and it can be later modified when visualizing
        # detections. In this case, the modified frame from one moment will be
        # input to the detection algorithm in the next moment, which will break
        # it. To avoid this, we return a copy of the frame.
        # TODO: Not sure what will happen in the unlikely case that this method 
        # is called between the moment the frame is read in the update method 
        # and the moment the new_frame flag is set to True.  
        if self.is_async:
            if not self.new_frame:
                return self.frame.copy()

            with self.lock:
                self.new_frame = False
                return self.frame
        else:
            (_, self.frame) = self.stream.read()
            return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        while self.thread is not None and self.thread.isAlive():
            pass
        self.stream.release()


class FrameServer(object):
    CAM_RESOLUTION = (1280, 960)
    SINGLE_WINDOW = (960, 720)
    DOUBLE_WINDOW = (576, 864)

    def __init__(self, devices, cal_files=[], stitch_file=None):

        assert len(devices) > 0

        # TODO: move these parameters somewhere
        self.overlap = 1
        self.blur_width = 10

        # Load calibration files.
        self.cals = []
        for cal_file in cal_files:
            if cal_file is None:
                cal = None
            else:
                with open(cal_file, 'r') as json_file:
                    camera_data = json.load(json_file)

                cal = dict()
                cal['mtx'] = np.array(camera_data["mtx"])
                cal['dist'] = np.array(camera_data["dist"])
                cal['new_mtx'], cal['roi'] = cv2.getOptimalNewCameraMatrix(cal['mtx'],
                                                                           cal['dist'],
                                                                           self.CAM_RESOLUTION,
                                                                           0,
                                                                           self.CAM_RESOLUTION)
            self.cals.append(cal)

        # Load stitching parameters file.
        self.warp_matrix = None
        self.crop_limits = None
        if stitch_file is not None:
            with open(stitch_file, 'r') as json_file:
                stitch_data = json.load(json_file)

            self.warp_matrix = np.array(stitch_data['tf_mat'])
            self.crop_limits = stitch_data['crop']
            self.frame_size = (stitch_data['dim'][1], stitch_data['dim'][0])
            self.join_frames = self.affine_join
        else:
            self.warp_matrix = None
            self.join_frames = self.fixed_join
            self.frame_size = (self.CAM_RESOLUTION[1], self.CAM_RESOLUTION[0])

        # Open camera streams.
        self.streams = []
        for dev in devices:
            vs = WebcamStream(dev, run_async=True)
            vs.start()
            self.streams.append(vs)

        self.window_size = self.SINGLE_WINDOW if len(devices) == 1 else self.DOUBLE_WINDOW

        # Set camera transformation parameters.
        self.resolution = 0.04 / 18  # meters per pixel

        time.sleep(2.0)

    def stop(self):
        for vs in self.streams:
            vs.stop()

    # TODO: rewrite this method to be more professional. (multiple returns)
    def grab(self, ret_original=False):
        original_frames = []
        for vs in self.streams:
            frame = vs.read()
            original_frames.append(frame)

        if self.cals:
            undistorted = self.undistort(original_frames)
            undistorted_joined = self.join_frames(undistorted)
            if ret_original:
                original_joined = self.join_frames(original_frames)
                return undistorted, undistorted_joined, original_frames, original_joined
            else:
                return undistorted, undistorted_joined

        original_joined = self.join_frames(original_frames)
        return original_frames, original_joined

    def undistort(self, frames):
        indv_frames = []
        for frame, cal in zip(frames, self.cals):
            if cal is None:
                indv_frames.append(frame)
            else:
                indv_frames.append(cv2.undistort(frame, cal['mtx'], cal['dist'], None, cal['new_mtx']))

        return indv_frames

    def scale(self, size):
        return size * self.resolution

    def transform(self, cam_pt):
        x_real = 0.0 + cam_pt[0] * self.resolution
        y_real = 0.0 + (self.frame_size[1] - cam_pt[1]) * self.resolution
        return (x_real, y_real)

    def inverse_transform(self, world_pt):
        raise NotImplementedError("Why do you need this?")

    def fixed_join(self, frames):
        cropped = []
        for i, frame in enumerate(frames):
            new_frame = frame.copy()
            if i != 0:
                # Crop upper part.
                new_frame = new_frame[self.overlap:, :]
            if i != len(frames) - 1:
                # Crop lower part.
                new_frame = new_frame[:-self.overlap, :]
            cropped.append(new_frame)

        return cv2.vconcat(cropped)

    def affine_join(self, frames):
        warp = cv2.warpAffine(frames[1], self.warp_matrix,
                              (frames[1].shape[1], frames[1].shape[0] + frames[1].shape[0]))
        warp[0:frames[0].shape[0], 0:frames[0].shape[1]] = frames[0]
        warp = warp[self.crop_limits[0][0]:self.crop_limits[0][1],
                    self.crop_limits[1][0]:self.crop_limits[1][1]]

        img_mid = frames[0].shape[0]
        roi = warp[img_mid - self.blur_width:img_mid + self.blur_width, :]
        blur = cv2.medianBlur(roi, 5)
        warp[img_mid - self.blur_width:img_mid + self.blur_width, :] = blur

        return warp


def main():
    cams = [
        '/dev/video4',
        '/dev/video2',
    ]

    calibrations = [
        '/home/marko/WS/sphero_ws/src/sphero/sphero_localization/config/camera_north.json',
        '/home/marko/WS/sphero_ws/src/sphero/sphero_localization/config/camera_south.json'
    ]
    stitch_param = '/home/marko/WS/sphero_ws/src/sphero/sphero_localization/config/stitch_params.json'

    # cv2.namedWindow("north", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("south", cv2.WINDOW_NORMAL)
    cv2.namedWindow("joined", cv2.WINDOW_NORMAL)

    fs = FrameServer(cams, calibrations, stitch_param)

    from collections import deque
    s = 0
    n = 20
    history = deque([0] * n, maxlen=n)
    iter_start = time.perf_counter()
    while True:
        grab_start = time.perf_counter()
        _, joined = fs.grab()
        iter_end = time.perf_counter()
        elapsed = iter_end - iter_start
        s += elapsed - history.popleft()
        history.append(elapsed)
        print(f"{1 / elapsed:5.2f} | avg = {1 / (s / n):5.2f} | {iter_end - grab_start}")
        iter_start = iter_end

        # cv2.imshow("north", individual[0])
        # cv2.resizeWindow('north', 960, 720)
        # cv2.imshow("south", individual[1])
        # cv2.resizeWindow('south', 960, 720)

        cv2.imshow("joined", joined)
        cv2.resizeWindow('joined', fs.window_size)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    fs.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
