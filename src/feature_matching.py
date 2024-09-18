import numpy as np

def match_features(desc_1,desc_2, ratio_threshold=0.8):
    mathces = []

    for i, desc1 in enumerate(desc_1):
        # compute the distance to all descriptors on second image
        distances =  np.linalg.norm(desc_2-desc1, axis=1)

        # Finf the best match
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        # Find the second best-match
        distances[best_idx] = np.inf  # Temporarily remove the best match
        second_best_distance = np.min(distances)

        # Apply lowe's ratio to filter ambigous matches
        if best_distance<ratio_threshold * second_best_distance:
            mathces.append((i, best_idx))
    
    return mathces

def match_features_bidirectional(descriptors1, descriptors2, ratio_threshold=0.75):
    matches = []
    
    for i, desc1 in enumerate(descriptors1):
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]
        
        distances[best_idx] = np.inf
        second_best_distance = np.min(distances)
        
        if best_distance < ratio_threshold * second_best_distance:
            # Check if the match is reciprocal
            desc2 = descriptors2[best_idx]
            distances_back = np.linalg.norm(descriptors1 - desc2, axis=1)
            back_best_idx = np.argmin(distances_back)
            if back_best_idx == i:
                matches.append((i, best_idx))
    
    return matches
