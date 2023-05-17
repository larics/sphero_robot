"""
This script will start a stream from two cameras and display them concatenated.
You should place three recognizible features in the part where images overlap.
This will be used for finding the transformation between the cameras.
When ready, press the 't' key to store the pictures on disk.
"""

import cv2
from datetime import datetime
from pathlib import Path
from sphero_localization.duo_c270 import FrameServer
import yaml

## Make necessary folder structure.
save_path = Path(__file__).parent / 'pictures/raw'
save_path.mkdir(parents=True, exist_ok=True)

## Load camera configuration files.
main_config_path = Path(__file__).parent / '../../config/config.yaml'
with open(main_config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
cams = config['cameras']
calibrations = [f"{main_config_path.rpartition('/')[0]}/{cal}" for cal in config['calibrations']]

## Start the stream.
cv2.namedWindow("joined", cv2.WINDOW_NORMAL)
fs = FrameServer(cams, calibrations)

while True:
    frames, joined = fs.grab()
    
    cv2.imshow("joined", joined)
    cv2.resizeWindow('joined', fs.window_size)

    key = cv2.waitKey(1) & 0xFF
    # If the 'q' key is pressed, stop the loop.
    if key == ord("q"):
        break
    # If the 't' key is pressed, stop the loop and stich the last available frame.
    if key == ord("t"):
        for i, frame in enumerate(frames):
            cv2.imwrite(f"{save_path}/calibrate_{i}.png", frame)
        break
    
fs.stop()
cv2.destroyAllWindows()
