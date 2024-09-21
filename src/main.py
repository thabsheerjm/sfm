#!/usr/bin/env python3
import data_loader
import feature_detection as fd
import feature_description as fp
import feature_matching as fm
import fundamental_matrix as fdm
import triangulation as tg
import visualization as vs

import matplotlib.pyplot as plt
import numpy as np

import os 
os.environ["NUMEXPR_MAX_THREADS"] = "24"


K = np.array([
    [2559.68, 0, 1536],
    [0, 2559.68, 1152],
    [0, 0, 1]
])


gray, images = data_loader.load_images('data/colmap_southB/images/')


points_3d_list = []
colors_list = []
camera_positions = []  

# Iterate over pairs of consecutive images
for i in range(len(gray)-1):
    image1 = gray[i]
    image2 = gray[i+1]
    color_image1 = images[i]
    
    # Detect keypoints and extract descriptors for both images(grayscale)
    keypoints1 = fd.harris_corner_detector(image1,window_size=5,k=0.03,threshold=5000)
    keypoints2 = fd.harris_corner_detector(image2,window_size=5,k=0.03,threshold=5000)

    descr1 = fp.extract_descriptors(image1,keypoints1,patch_size=9)
    descr2 = fp.extract_descriptors(image2,keypoints2,patch_size=9)

    # Match feature on two images
    matches = fm.match_features(descr1,descr2)

    #Estimate the fundamental matrix using RANSAC
    if len(matches) >= 8:
        F, inliers = fdm.ransac_F(matches, keypoints1, keypoints2)
    else:
        print(f"Not enough matches to compute the Fundamental Matrix between images {i} and {i+1}. Skipping...")
        continue

    # Essential matrix
    E = fdm.compute_essential_matrix(F, K)
    # print(f"Essential matrix: \n", E)

    # Decompose essential matrix into R,t
    R1, R2, t = fdm.decompose_essential_matrix(E)

    #Setup the projection matrices for the two camera views
    P1 = np.hstack((np.eye(3), np.zeros((3,1)))) # proj matrix for cam1 [I 0]
    
    P2_1 = np.hstack((R1, t.reshape(3,1))) #Proj matrix for cam2  using R1,t 
    P2_2 = np.hstack((R1, -t.reshape(3,1)))#Proj matrix for cam2  using R1,-t 
    P2_3 = np.hstack((R2, t.reshape(3,1))) #Proj matrix for cam2  using R2,t 
    P2_4 = np.hstack((R2, -t.reshape(3,1)))#Proj matrix for cam2  using R2,-t 
    

    # TRiangulate the 3D points using the four combos
    points_3d_1 = tg.triangulate_points(P1, P2_1, keypoints1, keypoints2, inliers)
    points_3d_2 = tg.triangulate_points(P1, P2_2, keypoints1, keypoints2, inliers)
    points_3d_3 = tg.triangulate_points(P1, P2_3, keypoints1, keypoints2, inliers)
    points_3d_4 = tg.triangulate_points(P1, P2_4, keypoints1, keypoints2, inliers)

    #check cheirality 
    if tg.check_cheirality(P1, P2_1, points_3d_1):
        correct_P2 = P2_1
        correct_points_3d = points_3d_1
        print("Using R1 and t for the correct camera pose")
    elif tg.check_cheirality(P1, P2_2, points_3d_2):
        correct_P2 = P2_2
        correct_points_3d = points_3d_2
        print("Using R1 and -t for the correct camera pose")
    elif tg.check_cheirality(P1, P2_3, points_3d_3):
        correct_P2 = P2_3
        correct_points_3d = points_3d_3
        print("Using R2 and t for the correct camera pose")
    elif tg.check_cheirality(P1, P2_4, points_3d_4):
        correct_P2 = P2_4
        correct_points_3d = points_3d_4
        print("Using R2 and -t for the correct camera pose")
    
    else:
        print("No valid solution found based on chierality")
        continue
    
    # Triangulate 3D points and assign color based on the correct projection matrix
    points_3d_list, colors_list = tg.triangulate_and_color(P1, correct_P2, keypoints1, keypoints2, inliers, color_image1, points_3d_list, colors_list)

    camera_positions.append(t)

vs.save_point_cloud_with_trajectory(points_3d_list, colors_list, camera_positions, filename="colored_point_cloud_with_trajectory.ply")



