import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Structure from motion")

    parser.add_argument('--image_dir',type=str,required=True, help='Directory containing images for resconstruction')
    parser.add_argument('--data_name',type=str,required=True, help='Name of the dataset')
    parser.add_argument('--output_dir', type=str, default='../results/pointcloud/', help='Directory to save output results')
    parser.add_argument('--focal_length', type=float, default=800.0, help='Camera focal length in pixels')
    parser.add_argument('--principal_point', type=float, nargs=2, default=[512.0, 384.0], help='Camera principal point (cx, cy)')   
    parser.add_argument('--k_harris', type=float, default=0.04, help='Harris detector free parameter')
    parser.add_argument('--threshold_harris', type=float, default=1e6, help='Threshold for Harris corner detection')
    parser.add_argument('--ransac_iterations', type=int, default=1000, help='Number of RANSAC iterations')
    parser.add_argument('--ransac_threshold', type=float, default=0.01, help='RANSAC inlier threshold')

    return parser.parse_args() 