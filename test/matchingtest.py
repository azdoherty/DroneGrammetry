import unittest
import cv2
from MergeImages import *
import time

class TestMatchingMethods(unittest.TestCase):
    def setUp(self):
        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 2)))

    def test_1(self):
        keypoint_extractor_type = "ORB"
        matcher_type = "FlannORB"
        img1 = cv2.imread('t1.jpg', 0)  # queryImage
        img2 = cv2.imread('t2.jpg', 0)  # trainImage
        kp1, dsc1 = get_keypoints(img1, searchtype=keypoint_extractor_type)
        kp2, dsc2 = get_keypoints(img2, searchtype=keypoint_extractor_type)
        gm = match_keypoints(dsc1, dsc2, matchertype=matcher_type)
        M, matchesMask = find_homography(kp1, kp2, gm)
        draw_kps(img1, img2, kp1, kp2, gm, matchesMask, fname="test_orb_flann.png")

    def fail_2(self):
        keypoint_extractor_type = "surf"
        matcher_type = "FlannSURF"
        img1 = cv2.imread('t1.jpg', 0)  # queryImage
        img2 = cv2.imread('t2.jpg', 0)  # trainImage
        kp1, dsc1 = get_keypoints(img1, searchtype=keypoint_extractor_type)
        kp2, dsc2 = get_keypoints(img2, searchtype=keypoint_extractor_type)
        gm = match_keypoints(dsc1, dsc2, matchertype=matcher_type)
        M, matchesMask = find_homography(kp1, kp2, gm)
        draw_kps(img1, img2, kp1, kp2, gm, matchesMask, fname="test_surf_flann.png")

if __name__ == '__main__':
    unittest.main()