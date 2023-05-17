"""
This script will load calibration images, detect three distinct keypoints, and
then, using their relative positions, calculate the transformation needed to 
bring these images together.

Currently, the detector in use is a simple blob detector working on filtered
brightness data. This means that provided keypoints in images should ideally be
Spheros with LEDs.
"""
import json
import numpy as np
import cv2
from pathlib import Path


def main():
    load_path = Path(__file__).parent / 'pictures/raw'
    
    cv_images = []
    images = load_path.glob('calibrate*.png')
    for img in sorted(images):
        cv_images.append(cv2.imread(str(img)))
        print(img.name)
    print(len(cv_images), "images found.")
    
    from sphero_localization.blob_detector import SpheroBlobDetector
    
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 200
    params.maxThreshold = 255
    # Filter by Area.
    params.filterByArea = False
    params.minArea = 400
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.2
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.5
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)
    
    detector = SpheroBlobDetector(params=params)
    kpts1, frame1, mask1 = detector.detect(cv_images[0])
    kpts2, frame2, mask2 = detector.detect(cv_images[1])
    cv2.imshow("frame1", frame1)
    cv2.imshow("frame2", frame2)
    cv2.resizeWindow("frame1", (576, 864 // 2))
    cv2.resizeWindow("frame2", (576, 864 // 2))
    
    srcTri = np.array([kpt.pt for kpt in kpts2]).astype(np.float32)
    dstTri = np.array([kpt.pt for kpt in kpts1]).astype(np.float32)

    warp_mat = cv2.getAffineTransform(srcTri, dstTri)

    # Overwrite mode
    warp2 = cv2.warpAffine(frame2, warp_mat, (frame2.shape[1], frame2.shape[0] + frame2.shape[0]))
    warp2[0:frame1.shape[0], 0:frame1.shape[1]] = frame1    
    
    # Blending mode
    # warp2 = cv2.warpAffine(frame2, warp_mat, (frame2.shape[1], frame2.shape[0] + frame2.shape[0]))
    # warp1 = np.zeros_like(warp2)
    # warp1[0:frame1.shape[0], 0:frame1.shape[1]] = frame1    
    # warp2 = cv2.addWeighted(warp1, 0.5, warp2, 0.5, 0.0)
    
    cv2.imshow("result", warp2)
    cv2.resizeWindow("result", (576, 864))
    
    # Crop
    crop_x = 0
    crop_y = 0
    crop_wait = True
    
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            nonlocal crop_x, crop_y, crop_wait
            crop_x, crop_y = x, y
            crop_wait = False
            print("done")
    
    cv2.setMouseCallback("result", on_mouse_click)
    
    while crop_wait:
        key = cv2.waitKey(1) & 0xFF
        # If the 'q' key is pressed, exit.
        if key == ord("q"):
            return
        
    cropped = warp2[0:crop_y, 0:crop_x]
    cv2.imshow("result", cropped)
    cv2.waitKey(0)
    
    final_img = dict(
        tf_mat=warp_mat.tolist(),
        crop=[[0, crop_y], [0, crop_x]],
        dim=cropped.shape
    )
    
    save_path = Path(__file__).parent / '../../config/stitch_params.json'
    with open(save_path, "w") as f:
        json.dump(final_img, f, indent=4)
        
    cv2.imwrite(Path(__file__).parent / 'pictures/stitched.png', cropped)
        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()