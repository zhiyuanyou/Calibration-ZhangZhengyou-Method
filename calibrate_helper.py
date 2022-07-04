# -*- coding: utf-8 -*-
"""
Calibrate the Camera with Zhang Zhengyou Method.

By You Zhiyuan, 2022.07.04, zhiyuanyou@foxmail.com
"""

import os
import glob

import cv2
import numpy as np


class Calibrator(object):
    def __init__(self, img_dir, shape_inner_corner, size_grid, visualization=True):
        """
        --parameters--
        img_dir: the directory that save images for calibration, str
        shape_inner_corner: the shape of inner corner, Array of int, (h, w)
        size_grid: the real size of a grid in calibrator, float
        visualization: whether visualization, bool
        """
        self.img_dir = img_dir
        self.shape_inner_corner = shape_inner_corner
        self.size_grid = size_grid
        self.visualization = visualization
        self.mat_intri = None # intrinsic matrix
        self.coff_dis = None # cofficients of distortion

        # create the conner in world space
        w, h = shape_inner_corner
        # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
        # like (0,0,0), (1,0,0), (2,0,0) ...., (10,7,0)
        cp_int = np.zeros((w * h, 3), np.float32)
        cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        # cp_world: corner point in world space, save the coordinate of corner points in world space
        self.cp_world = cp_int * size_grid

        # images
        self.img_paths = []
        for extension in ["jpg", "png", "jpeg"]:
            self.img_paths += glob.glob(os.path.join(img_dir, "*.{}".format(extension)))
        assert len(self.img_paths), "No images for calibration found!"


    def calibrate_camera(self):
        w, h = self.shape_inner_corner
        # criteria: only for subpix calibration, which is not used here
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        points_world = [] # the points in world space
        points_pixel = [] # the points in pixel space (relevant to points_world)
        for img_path in self.img_paths:
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find the corners, cp_img: corner points in pixel space
            ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None)
            # if ret is True, save
            if ret:
                # cv2.cornerSubPix(gray_img, cp_img, (11,11), (-1,-1), criteria)
                points_world.append(self.cp_world)
                points_pixel.append(cp_img)
                # view the corners
                if self.visualization:
                    cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
                    cv2.imshow('FoundCorners', img)
                    cv2.waitKey(500)

        # calibrate the camera
        ret, mat_intri, coff_dis, v_rot, v_trans = cv2.calibrateCamera(points_world, points_pixel, gray_img.shape[::-1], None, None)
        print ("ret: {}".format(ret))
        print ("intrinsic matrix: \n {}".format(mat_intri))
        # in the form of (k_1, k_2, p_1, p_2, k_3)
        print ("distortion cofficients: \n {}".format(coff_dis))
        print ("rotation vectors: \n {}".format(v_rot))
        print ("translation vectors: \n {}".format(v_trans))

        # calculate the error of reproject
        total_error = 0
        for i in range(len(points_world)):
            points_pixel_repro, _ = cv2.projectPoints(points_world[i], v_rot[i], v_trans[i], mat_intri, coff_dis)
            error = cv2.norm(points_pixel[i], points_pixel_repro, cv2.NORM_L2) / len(points_pixel_repro)
            total_error += error
        print("Average error of reproject: {}".format(total_error / len(points_world)))

        self.mat_intri = mat_intri
        self.coff_dis = coff_dis
        return mat_intri, coff_dis


    def dedistortion(self, save_dir):
        # if not calibrated, calibrate first
        if self.mat_intri is None:
            assert self.coff_dis is None
            self.calibrate_camera()

        w, h = self.shape_inner_corner
        for img_path in self.img_paths:
            _, img_name = os.path.split(img_path)
            img = cv2.imread(img_path)
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mat_intri, self.coff_dis, (w,h), 0, (w,h))
            dst = cv2.undistort(img, self.mat_intri, self.coff_dis, None, newcameramtx)
            # clip the image
            # x, y, w, h = roi
            # dst = dst[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(save_dir, img_name), dst)
        print("Dedistorted images have been saved to: {}".format(save_dir))
