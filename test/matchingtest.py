import unittest
import cv2
from MergeImages import *

class TestMatchingMethods(unittest.TestCase):


    def test_1(self):
        img1 = cv2.imread('t1.jpg', 0)  # queryImage
        img2 = cv2.imread('t2.jpg', 0)  # trainImage
        kp1, dsc1 = get_keypoints(img1)
        kp2, dsc2 = get_keypoints(img2)
        gm = match_keypoints(dsc1, dsc2)
        M, matchesMask = find_homography(kp1, kp2, gm)
        draw_kps(img1, img2, kp1, kp2, gm, matchesMask, fname="test1.png")


if __name__ == '__main__':
    unittest.main()