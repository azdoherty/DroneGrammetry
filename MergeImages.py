import cv2
import numpy as np
import os
import sys


def draw_matches(img1, kp1, img2, kp2, matches, color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    md = 0
    for m in matches:
        if m.distance > md:
            md = m.distance
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = np.random.randint(0, 256, 3) if len(img1.shape) == 3 else np.random.randint(0, 256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        c = [int(c[0]), int(c[1]), int(c[2])]
        #c = int(255*m.distance/md)
        print(c)
        #c = [255,255,255]

        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)

    return new_img

def stitch_images(imlist, outfolder=None):
    """
    :param imlist: list of loaded image objects
    :return: single image of all of them mapped together
    #TODO - add use of neighbor list to only match each image to its neighbors
          - add bundle adjustment
          - add 3d reconstruction
    """
    kpd = cv2.ORB_create()
    kps = []
    descs = []
    for im in imlist:
        # extract keypoints
        kp, des = kpd.detectAndCompute(im, None)
        kps.append(kp)
        descs.append(des)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    match_array = np.zeros((len(imlist), len(imlist)))
    matchds = np.zeros((len(imlist), len(imlist)))
    for i in range(len(imlist)):
        # find closest matches in series
        for j in range(1 + i, len(imlist)):
            print(i,j)
            matches = bf.match(descs[i], descs[j])
            matches = sorted(matches, key=lambda x: x.distance)
            #evaluate_match_quality(matches)
            check = draw_matches(imlist[i], kps[i], imlist[j], kps[j], matches[:50])
            src_pts = np.float32([kps[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kps[j][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w, c = imlist[i].shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            out = stitch2(M, imlist[i], imlist[j])
            cv2.imwrite(os.path.join(outfolder, "{}_{}.png".format(i,j)), out)

def stitch2(M, im1, im2):
    inv_M = np.linalg.inv(M)
    f1 = np.dot(inv_M, np.array([0, 0, 1]))
    f1 = f1 / f1[-1]
    inv_M[0][-1] += abs(f1[0])
    inv_M[1][-1] += abs(f1[1])
    ds = np.dot(inv_M, np.array([im1.shape[1], im1.shape[0], 1]))
    offsety = abs(int(f1[1]))
    offsetx = abs(int(f1[0]))
    dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
    tmp = cv2.warpPerspective(im1, inv_M, dsize)
    tmp[offsety:im2.shape[0]+offsety, offsetx:im2.shape[1]+offsetx] = im2
    return tmp



def evaluate_match_quality(matches, thresh=.7):
    """
    evaluate quality of knn matches
    :param matches:
    :param thresh:
    :return:
    """
    marr = np.zeros((len(matches), 2))
    quality = np.zeros((len(matches), 1))
    for i in range(len(matches)):
        marr[i,0] = matches[i][0].distance
        marr[i,1] = matches[i][1].distance
        quality[i] = marr[i, 0] < thresh*marr[i, 1]


def load_image_dir(imdir, converbw = True):
    flist = os.listdir(imdir)
    imlist = []
    for im in flist:
        imlist.append(cv2.imread(os.path.join(imdir, im)))
        #imlist[-1] = cv2.cvtColor(imlist[-1], cv2.COLOR_RGB2GRAY)

    return imlist

def load_n_stitch(imdir, **kwargs):
    imlist = load_image_dir(imdir)
    if "output" in kwargs:
        stitched = stitch_images(imlist, outfolder=kwargs["output"])
    else:
        stitched = stitch_images(imlist)
    return stitched




if __name__ == "__main__":
    dir = sys.argv[1]
    stitched = load_n_stitch(dir, output="testout")
    fname = os.path.join(dir, "stitched.png")
    #cv2.imwrite(fname, stitched)

