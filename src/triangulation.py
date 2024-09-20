import cupy as cp 

def triangulate_point(p1,p2,x1,x2):
    """
    Triangulate a single 3D point from two views.
    
    Args:
        P1: Projection matrix for the first camera (3x4).
        P2: Projection matrix for the second camera (3x4).
        x1: 2D point in the first image (in homogeneous coordinates).
        x2: 2D point in the second image (in homogeneous coordinates).
    
    Returns:
        X: The triangulated 3D point (in homogeneous coordinates).
    """

    A = cp.zeros((4,4))

    #Build the linear sys of equations
    A[0] = x1[0] * p1[2] - p1[0]
    A[1] = x1[1] * p1[2] - p1[1]
    A[2] = x2[0] * p2[2] - p2[0]
    A[3] = x2[1] * p2[2] - p2[1]

    #solve for x
    _,_,Vt = cp.linalg.svd(A)
    x = Vt[-1]
    x = x / x[3] # convert to non-homogeneous coordinates
    return x[:3]


def triangulate_points(p1,p2,keypoints1,keypoints2,matches):
    """
    Triangulate 3D points from two views.
    
    Args:
        P1: Projection matrix for the first camera (3x4).
        P2: Projection matrix for the second camera (3x4).
        keypoints1: List of keypoints in the first image.
        keypoints2: List of keypoints in the second image.
        matches: List of matched keypoint indices.
    
    Returns:
        points_3d: List of triangulated 3D points.
    """

    points_3d = []

    for (i,j) in matches:
        x1 = cp.array([keypoints1[i][0], keypoints1[i][1],1]) #homogeneous coordinates
        x2 = cp.array([keypoints2[j][0], keypoints2[j][1],1])

        # TRiangulate the 3D point
        x = triangulate_point(p1,p2,x1,x2)
        points_3d.append(x)

    return cp.array(points_3d)


def triangulate_and_color(P1, P2, keypoints1, keypoints2, matches, color_image, points_3d_list, colors_list):
    for (i, j) in matches:
        x1 = cp.array([keypoints1[i][0], keypoints1[i][1], 1])
        x2 = cp.array([keypoints2[j][0], keypoints2[j][1], 1])

        # Triangulate the 3D point
        X = triangulate_point(P1, P2, x1, x2)
        
        # Only keep valid points (positive depth)
        if X[2] > 0:
            points_3d_list.append(X)

            # Assign color to the 3D point using the first image's color data
            color = color_image[int(keypoints1[i][1]), int(keypoints1[i][0])]
            colors_list.append(color / 255.0)  # Normalize color values to [0, 1]

    return points_3d_list, colors_list


def check_cheirality(P1, P2, points_3d):
    """
    Check the cheirality condition for triangulated 3D points.
    
    Args:
        P1: Projection matrix for the first camera.
        P2: Projection matrix for the second camera.
        points_3d: Triangulated 3D points.
    
    Returns:
        bool: True if the majority of points have positive depth in both views.
    """
     
    # project points onto to the cam1
    points1 = P1 @ cp.hstack((points_3d, cp.ones((points_3d.shape[0], 1)))).T
    points1 = points1.T
    depth1 = points1[:,2] # Z-coordinate

    # project points onto to the cam1
    points2 = P2 @ cp.hstack((points_3d, cp.ones((points_3d.shape[0], 1)))).T
    points2 = points2.T
    depth2 = points2[:,2]

    # check if the majority of points have positive depth in both views
    return cp.sum(depth1>0)>(0.5*len(depth1)) and cp.sum(depth2>0)>(0.5*len(depth2))




