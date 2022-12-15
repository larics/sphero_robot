#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import time
import json
import numpy as np
from imutils.video import VideoStream

# TODO: transform
# TODO: threading


class FrameServer(object):
    CAM_RESOLUTION = (1280, 960)
    SINGLE_WINDOW = (960, 720)
    DOUBLE_WINDOW = (576, 864)
    
    def __init__(self, devices, cal_files=[]):
        
        assert len(devices) > 0
        
        self.overlap = 1
        
        self.streams = []
        self.cals = []
        
        # Open camera streams.
        for dev in devices:
            vs = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            vs.set(cv2.CAP_PROP_FPS, 30)
            print(f"Camera frame rate set to {vs.get(cv2.CAP_PROP_FPS)}.")
            self.streams.append(vs)
        
        self.window_size = self.SINGLE_WINDOW if len(devices) == 1 else self.DOUBLE_WINDOW
            
        # Open calibration files.
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
            
        # Set camera transformation parameters.
        self.resolution = 0.001875
            
        time.sleep(2.0)      
            
    def stop(self):
        for vs in self.streams:
            vs.release()
            
    # TODO: rewrite this method to be more professional. (multiple returns)
    def grab(self, ret_original=False):
        original_frames = []
        for vs in self.streams:
            ret, frame = vs.read()
            original_frames.append(frame)
            
        if self.cals:
            undistorted = self.undistort(original_frames)
            undistorted_joined = FrameServer.fixed_join(undistorted, self.overlap)
            if ret_original:
                original_joined = FrameServer.fixed_join(original_frames, self.overlap)
                return undistorted, undistorted_joined, original_frames, original_joined
            else:
                return undistorted, undistorted_joined
        
        original_joined = FrameServer.fixed_join(original_frames, self.overlap)
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
        x_real = 0.0 + -cam_pt[0] * self.resolution
        y_real = 0.0 + cam_pt[1] * self.resolution
        return (x_real, y_real)
    
    def inverse_transform(self, world_pt):
        x_cam = -world_pt[0] / self.resolution
        y_cam = world_pt[1] / self.resolution
        return (x_cam, y_cam)
    
    @staticmethod
    def fixed_join(frames, overlap):
        offset = overlap
        
        cropped = []
        for i, frame in enumerate(frames):
            new_frame = frame.copy()
            if i != 0:
                # Crop upper part.
                new_frame = new_frame[offset:, :]
            if i != len(frames) - 1:
                # Crop lower part.
                new_frame = new_frame[:-offset, :]
            cropped.append(new_frame)
        
        return cv2.vconcat(cropped)
            

def main():
    cams = [
            # '/dev/video4',
            '/dev/video2',
    ]
    
    calibrations = [
        # '/home/marko/WS/sphero_ws/src/sphero/sphero_localization/config/camera_north.json',
        '/home/marko/WS/sphero_ws/src/sphero/sphero_localization/config/camera_south.json'
    ]
    
    # cv2.namedWindow("north", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("south", cv2.WINDOW_NORMAL)
    cv2.namedWindow("joined", cv2.WINDOW_NORMAL)
    
    fs = FrameServer(cams, calibrations)
    
    old = time.perf_counter()
    
    while True:
        _, joined = fs.grab()
        new = time.perf_counter()
        print(1 / (new - old))
        old = new
       
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