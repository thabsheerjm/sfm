import cupy as cp 
import random

def estimate_fundamental_matrix(matches, key1, key2):
    """
    Estimate the fundamental matrix using the 8-point algorithm.
    
    Args:
        matches: List of matched keypoint indices.
        keypoints1, keypoints2: Lists of keypoints from both images.
    
    Returns:
        F: Estimated 3x3 fundamental matrix.
    """
    A = []
    for (i,j) in matches:
        x1, y1 = key1[i]
        x2, y2 = key2[j]
        A.append([x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,1])
    
    A = cp.array(A)

    # SVD on A
    U, S, Vt = cp.linalg.svd(A)
    F = Vt[-1].reshape(3,3)

    #Enforce rank=2 constraint by setting smallest singular value to zero
    U, S, Vt = cp.linalg.svd(F)
    S[2]=0
    F = U @ cp.diag(S) @ Vt

    return F

def ransac_F(matches,key1,key2,num_iterations=1e3,threshold=1e-2):
    """
    Estimate the fundamental matrix using RANSAC to filter outliers.
    
    Args:
        matches: List of matched keypoint indices.
        keypoints1, keypoints2: Lists of keypoints from both images.
        num_iterations: Number of RANSAC iterations.
        threshold: Epipolar constraint threshold to consider a match as an inlier.
    
    Returns:
        best_F: The best estimated fundamental matrix.
        inliers: List of inlier matches.
    """
    best_F = None
    best_inliers = []
    max_inliers = 0

    for _ in range(int(num_iterations)):
        # Randomly choose 8 matches
        sample_matches = random.sample(matches,8)

        # Estimate F 
        F = estimate_fundamental_matrix(sample_matches, key1, key2)
        inliers = []
        for (i,j) in matches:
            x1,y1 = key1[i]
            x2,y2 = key2[j]
            x1_h = cp.array([x1,y1,1])
            x2_h = cp.array([x2,y2,1])

            # Epipolar constraint error: x2.T * F * x1 should be close to 0
            error = cp.abs(x2_h.T @ F @ x1_h)
            if error< threshold:
                inliers.append((i,j))

        # update best if more inliers are found
        if len(inliers)>max_inliers:
            max_inliers =len(inliers)
            best_inliers = inliers
            best_F = F

    return best_F, best_inliers


def compute_essential_matrix(F, K):
    """
    Compute the Essential Matrix from the Fundamental Matrix and camera intrinsic matrix.
    
    Args:
        F: Fundamental matrix (3x3).
        K: Camera intrinsic matrix (3x3).
    
    Returns:
        E: Essential matrix (3x3).
    """
    return K.T @ F @ K


def decompose_essential_matrix(E):
    """
    Decompose the Essential Matrix into possible rotation matrices and translation vector.
    
    Args:
        E: Essential matrix (3x3).
    
    Returns:
        R1, R2: Two possible rotation matrices (3x3).
        t: Translation vector (3x1).
    """

    # perform svd on E
    U, S, Vt = cp.linalg.svd(E)

    #Ensure that the determinant of u and Vt is positive (to avoid reflections)
    if cp.linalg.det(U) < 0:
        U *= -1
    if cp.linalg.det(Vt) < 0:
        Vt *= -1

    # W is special matrix used to create the two posiible rotation matrices
    W = cp.array([[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 1]])
    
    # Possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    #Translation vector
    t = U[:,2]

    return R1, R2, t




            



