import open3d as o3d
import numpy as np
import copy

def load_point_cloud(filename):
    """Load a PCD file into an Open3D point cloud object."""
    pcd = o3d.io.read_point_cloud(filename)
    return pcd

def apply_icp(source_pcd, target_pcd, threshold=1.0, trans_init=None):
    """Align two point cloud objects using the ICP algorithm."""
    if trans_init is None:
        trans_init = np.identity(4)  # Initial transformation
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000000)
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria)
    return result.transformation

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.015,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])




def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down, 
        target=target_down, 
        source_feature=source_fpfh, 
        target_feature=target_fpfh, 
        mutual_filter=True, 
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(), 
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def remove_edge_degrees(pcd, cut_degree, cut_side='leftmost'):
    # Convert point cloud to a NumPy array
    xyz = np.asarray(pcd.points)
    
    # Calculate azimuth angle in radians for each point
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])  # y, x for horizontal plane
    
    # Normalize angles to range from -π to π
    theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi

    # Convert cut_degree to radians
    cut_rad = np.deg2rad(cut_degree)

    # Determine the range to exclude based on the specified side
    if cut_side == 'rightmost':
        # For leftmost, exclude from the start angle to start angle + cut_rad
        exclude_min_angle_rad = 0
        exclude_max_angle_rad = cut_rad
    elif cut_side == 'leftmost':
        # For rightmost, exclude from end angle - cut_rad to end angle
        exclude_min_angle_rad = np.pi - cut_rad
        exclude_max_angle_rad = np.pi
    else:
        raise ValueError("cut_side must be 'leftmost' or 'rightmost'")

    # Correcting the logic based on the azimuth angle orientation
    # Keep points outside the exclusion range
    if cut_side == 'leftmost':
        mask = (theta < exclude_min_angle_rad) | (theta > exclude_max_angle_rad)
    else:  
        mask = (theta < exclude_min_angle_rad) | (theta > exclude_max_angle_rad)

    kept_xyz = xyz[mask]
    
    # Create a new point cloud from the kept points
    kept_pcd = o3d.geometry.PointCloud()
    kept_pcd.points = o3d.utility.Vector3dVector(kept_xyz)
    
    return kept_pcd


if __name__ == "__main__":
    # # Load your point cloud files
    source_filename = '/home/wyd/sensor_ws/ros2/calib_Feb28_40_indoor/lidar1/1.pcd'
    target_filename = '/home/wyd/sensor_ws/ros2/calib_Feb28_40_indoor/lidar2/1.pcd'
    
    source_pcd = load_point_cloud(source_filename)
    target_pcd = load_point_cloud(target_filename)

    # croped_source_pcd = remove_edge_degrees(source_pcd, 80, "leftmost")
    # croped_target_pcd = remove_edge_degrees(target_pcd, 60, "rightmost")
    # # o3d.visualization.draw_geometries([croped_target_pcd],
    # #                                   zoom=0.015,
    # #                                   front=[0.9288, -0.2951, -0.2242],
    # #                                   lookat=[1.6784, 2.0612, 1.4451],
    # #                                   up=[-0.3402, -0.9189, -0.1996])
    # # Example usage
    # voxel_size = 0.05  # depends on your dataset
    # source_down, source_fpfh = preprocess_point_cloud(croped_source_pcd, voxel_size)
    # target_down, target_fpfh = preprocess_point_cloud(croped_target_pcd, voxel_size)

    # result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    # print(result_ransac)


    # # Apply ICP
    # threshold = 0.5  # Maximum distance between corresponding points
    # transformation = apply_icp(croped_source_pcd, croped_target_pcd, threshold, trans_init=result_ransac.transformation)
    # np.savetxt("matrix_calib_Feb28_40_indoor.txt", transformation, fmt='%f')
    
    # # Visualize the result
    # draw_registration_result(croped_source_pcd, croped_target_pcd, transformation)
    transformation_matrix = np.loadtxt("matrix_calib_Feb28_40_indoor.txt")
    draw_registration_result(source_pcd, target_pcd, transformation_matrix)

    

