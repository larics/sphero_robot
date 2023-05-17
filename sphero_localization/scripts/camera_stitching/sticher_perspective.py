# import the necessary packages
import numpy as np
import cv2
from pathlib import Path


def stitch(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
    # Unpack the images, then detect keypoints and extract local invariant descriptors from them.
    (image1, image2) = images
    (kps1, features1) = detectAndDescribe(image1)
    (kps2, features2) = detectAndDescribe(image2)
    
    # Match features between the two images.
    M = matchKeypoints(kps1, kps2, features1, features2, ratio, reprojThresh)
    
    if M is None:
        return None

    # Apply a perspective warp to stitch the images together.
    (matches, H, status) = M
    result = cv2.warpPerspective(image2, H, (image2.shape[1], image2.shape[0] + image1.shape[0]))
    # result = cv2.warpAffine(image2, H, (image2.shape[1], image2.shape[0] + image1.shape[0]))
    result[0:image1.shape[0], 0:image1.shape[1]] = image1
    
    if showMatches:
        vis = drawMatches(image1, image2, kps1, kps2, matches, status)
        return (result, vis)
    return result


def detectAndDescribe(image):
    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.SIFT_create()
    (kps, features) = detector.detectAndCompute(gray, None)
    
    # Convert the keypoints from KeyPoint objects to NumPy arrays.
    kps = np.float32([kp.pt for kp in kps])
    
    return (kps, features)


def matchKeypoints(kps1, kps2, features1, features2, ratio, reprojThresh):
    # Compute the raw matches and initialize the list of actual matches.
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(features1, features2, 2)
    matches = []
    
    # Loop over the raw matches.
    for m in rawMatches:
        # Ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test).
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # Computing a homography requires at least 4 matches.
    if len(matches) > 4:
        # Construct the two sets of points.
        pts1 = np.float32([kps1[i] for (_, i) in matches])
        pts2 = np.float32([kps2[i] for (i, _) in matches])
        # Compute the homography between the two sets of points.
        (H, status) = cv2.findHomography(pts2, pts1, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    
    return None


def drawMatches(image1, image2, kps1, kps2, matches, status):
    # Initialize the output visualization image
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    vis = np.zeros((h1 + h2, max(w1, w2), 3), dtype="uint8")
    vis[0:h1:, 0:w1] = image1
    vis[h1:, 0:w2] = image2
    
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # Only process the match if the keypoint was successfully matched.
        if s == 1:
            ptA = (int(kps1[queryIdx][0]), int(kps1[queryIdx][1]))
            ptB = (int(kps2[trainIdx][0]), int(kps2[trainIdx][1]) + h1)
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    return vis


def main():
    load_path = Path(__file__).parent / 'pictures/raw'
    
    cv_images = []
    images = load_path.glob('calibrate*.png')
    for img in sorted(images):
        cv_images.append(cv2.imread(str(img)))
        print(img.name)
    print(len(cv_images), "images found.")
        
    cv2.namedWindow("matches", cv2.WINDOW_NORMAL)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    
    (result, vis) = stitch(cv_images, showMatches=True)
    
    cv2.imshow("matches", vis)
    cv2.imshow("result", result)
    
    cv2.resizeWindow("matches", (576, 864))
    cv2.resizeWindow("result", (576, 864))
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        # If the 'q' key is pressed, exit.
        if key == ord("q"):
            return


if __name__ == '__main__':
    main()