# -*- coding: utf-8 -*-
"""
Calibrate the Camera with Zhang Zhengyou Method.
Picture File Folder: "./pic/IR_camera_calib_img/", With Distortion. 

By You Zhiyuan, 2022.07.04, zhiyuanyou@foxmail.com
"""

import os

from calibrate_helper import Calibrator


def main():
    img_dir = "./pic/IR_camera_calib_img"
    shape_inner_corner = (11, 8)
    size_grid = 0.02
    # create calibrator
    calibrator = Calibrator(img_dir, shape_inner_corner, size_grid)
    # calibrate the camera
    mat_intri, coff_dis = calibrator.calibrate_camera()
    # dedistort and save the dedistortion result
    save_dir = "./pic/IR_dedistortion"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    calibrator.dedistortion(save_dir)


if __name__ == '__main__':
    main()
