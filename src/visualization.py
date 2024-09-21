
import numpy as np
import os
import matplotlib.pyplot as plt
from open3d import geometry, io, visualization, utility

def plot_3d_points(points_3d):
    """
    Plot the triangulated 3D points in a 3D scatter plot.
    
    Args:
        points_3d: Array of 3D points (Nx3).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def save_point_cloud(points_3d, colors, filename="output_cloud.ply"):
    points = np.array(points_3d)
    colors = np.array(colors)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(base_dir, '../results/pointcloud/')

    # Create Open3D point cloud object
    point_cloud = geometry.PointCloud()
    point_cloud.points = utility.Vector3dVector(points)
    point_cloud.colors = utility.Vector3dVector(colors)

    io.write_point_cloud(str(dir+filename), point_cloud)


def save_point_cloud_with_trajectory(points_3d, colors, camera_positions, filename="output_with_trajectory.ply"):
    points = np.array(points_3d)
    colors = np.array(colors)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(base_dir, '../results/pointcloud/')
    # Create an Open3D point cloud object
    point_cloud = geometry.PointCloud()

    # Convert points and colors to the correct Open3D format
    point_cloud.points = utility.Vector3dVector(points)
    point_cloud.colors = utility.Vector3dVector(colors)

    # Create a lineset for the camera trajectory
    trajectory_points = np.array(camera_positions)
    trajectory_lines = [[i, i+1] for i in range(len(camera_positions)-1)]  # Create lines between consecutive camera positions
    trajectory_color = [[1, 0, 0] for _ in range(len(trajectory_lines))]   # Red color for the camera traj

    line_set = geometry.LineSet()
    line_set.points = utility.Vector3dVector(trajectory_points)
    line_set.lines = utility.Vector2iVector(trajectory_lines)
    line_set.colors = utility.Vector3dVector(trajectory_color)

    # Save the point cloud and camera trajectory
    io.write_point_cloud(str(dir+filename), point_cloud)

    # Visualize point cloud and camera trajectory
    visualization.draw_geometries([point_cloud, line_set])


def visualize_point_cloud(points_3d, colors):
    points = np.array(points_3d)
    colors = np.array(colors)

    # Create Open3D point cloud object
    point_cloud = geometry.PointCloud()
    point_cloud.points = utility.Vector3dVector(points)
    point_cloud.colors = utility.Vector3dVector(colors)

    # Visualize point cloud
    visualization.draw_geometries([point_cloud])