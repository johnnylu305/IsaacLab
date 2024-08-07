import os
import glob
import numpy as np
import argparse
import pickle
import open3d as o3d
from collections import deque



# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to fill occupancy grid")
parser.add_argument("input", type=str, help="The root to the input occupancy grid.")
# parse the arguments
args_cli = parser.parse_args()


def add_neighbors(occupied_voxels, grid_shape):
    # Directions for 6-connected neighbors in a 3D grid
    directions = [ (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                     (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                     (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                     (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
                     (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                     (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

    x_dim, y_dim, z_dim = grid_shape
    neighbors = set()

    for x, y, z in occupied_voxels:
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < x_dim and 0 <= ny < y_dim and 0 <= nz < z_dim:
                neighbors.add((nx, ny, nz))
    
    # Combine original occupied voxels with their neighbors
    occupied_voxels.update(neighbors)
    
    return occupied_voxels


def find_occupied_voxels(grid):
    # Ensure grid is a numpy array
    grid = np.array(grid)
    x_dim, y_dim, z_dim = grid.shape
    
    # Directions for 6-connected neighbors in a 3D grid
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    
    # BFS to mark all reachable empty voxels from the exterior
    def bfs(start):
        queue = deque([start])
        visited = set()
        visited.add(start)
        
        while queue:
            x, y, z = queue.popleft()
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < x_dim and 0 <= ny < y_dim and 0 <= nz < z_dim:
                    if grid[nx, ny, nz] == 0 and (nx, ny, nz) not in visited:
                        visited.add((nx, ny, nz))
                        queue.append((nx, ny, nz))
        
        return visited
    
    # Start BFS from an exterior voxel
    exterior_empty_voxels = bfs((x_dim-1, y_dim-1, z_dim-1))
    
    occupied_voxels = set()
    
    # Collect all occupied voxels and mark all non-reachable empty voxels as occupied
    for z in range(z_dim):
        for x in range(x_dim):
            for y in range(y_dim):
                if grid[x, y, z] == 1 or (grid[x, y, z] == 0 and (x, y, z) not in exterior_empty_voxels):
                    #occupied_voxels.add((z, x, y))
                    # new version
                    occupied_voxels.add((x, y, z))
    
    return occupied_voxels

def fill_grid(path):
    hollow_occ = np.load(path)
    # x,y,z -> z,x,y
    hollow_occ = np.where(hollow_occ==2, 1, 0)#.transpose((2, 0, 1))
    #print(hollow_occ[0])
    #exit()
    #print(hollow_occ.shape, np.sum(hollow_occ), hollow_occ[0, 0, 0])
    x_len, y_len, z_len = hollow_occ.shape
    assert hollow_occ[-1, -1, -1] == 0
    occ_set = find_occupied_voxels(hollow_occ)
    print(np.sum(hollow_occ), len(occ_set))
    occ_set_dilated = add_neighbors(occ_set.copy(), hollow_occ.shape)
    print("set", len(occ_set), len(occ_set_dilated))
    output = os.path.join(os.path.split(path)[0], "fill_occ_set.pkl")
    # Save the occupied voxels to a file
    with open(output, 'wb') as file:
        pickle.dump(occ_set_dilated, file)
    return occ_set, z_len, x_len, y_len

def is_hollow(x, y, z, occ_grid, min_bound, max_bound):
    # List of the 8 surrounding voxel coordinates
    surrounding_voxels = [
        (x-1, y-1, z-1), (x-1, y-1, z+1), (x-1, y+1, z-1), (x-1, y+1, z+1),
        (x+1, y-1, z-1), (x+1, y-1, z+1), (x+1, y+1, z-1), (x+1, y+1, z+1)
    ]
    
    # Check if all surrounding voxels are occupied and within bounds
    for vx, vy, vz in surrounding_voxels:
        if not (min_bound[0] <= vx <= max_bound[0] and
                min_bound[1] <= vy <= max_bound[1] and
                min_bound[2] <= vz <= max_bound[2]):
            return False
        if (vx, vy, vz) not in occ_grid:
            return False
    
    return True

def convert_to_hollow(occ_grid, min_bound, max_bound):
    hollowed_grid = set()
    
    for voxel in occ_grid:
        x, y, z = voxel
        if not is_hollow(x, y, z, occ_grid, min_bound, max_bound):
            hollowed_grid.add(voxel)
    
    return hollowed_grid

def map_to_3d_grid(hollow_set, min_bound, max_bound, path):
    # Calculate the dimensions of the grid
    x_dim = max_bound[0] - min_bound[0] 
    y_dim = max_bound[1] - min_bound[1] 
    z_dim = max_bound[2] - min_bound[2] 
    
    # Initialize the grid with zeros
    grid = np.zeros((x_dim, y_dim, z_dim), dtype=int)
    
    # Map the hollow set to the grid
    for (x, y, z) in hollow_set:
        grid[x - min_bound[0], y - min_bound[1], z - min_bound[2]] = 1

    output = os.path.join(os.path.split(path)[0], "hollow_occ.npy")
    np.save(output, grid)
    print("hollow", np.sum(grid))
    return grid

def save_face(scene_path, occ_set):
    hollow_occ = np.load(scene_path)
    # x,y,z -> z,x,y
    hollow_occ = np.where(hollow_occ == 2, 1, 0)#.transpose((2, 0, 1))
    
    X, Y, Z = hollow_occ.shape
    face_visibility = np.zeros((X, Y, Z, 6), dtype=bool)
    
    faces = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    
    def is_visible(x, y, z, dx, dy, dz):
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nz < Z and 0 <= nx < X and 0 <= ny < Y:
            return (nx, ny, nz) not in occ_set
        return True
    
    for x, y, z in occ_set:
        if hollow_occ[x, y, z] == 1:
            for n, (dx, dy, dz) in enumerate(faces):
                face_visibility[x, y, z, n] = is_visible(x, y, z, dx, dy, dz)

    face_visibility = face_visibility#.transpose((1, 2, 0, 3))
    output = os.path.join(os.path.split(scene_path)[0], "faces.npy")
    np.save(output, face_visibility)
    return face_visibility

def vis_face_and_voxel(scene_path, face_vis):
    # Load the hollow_occ array
    hollow_occ = np.load(scene_path)
    hollow_occ = np.where(hollow_occ == 2, 1, 0)
    
    X, Y, Z = hollow_occ.shape
    faces = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    
    voxel_list = []
    normal_list = []

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if hollow_occ[x, y, z] == 1:
                    voxel_list.append([x, y, z])
                    for n, (dx, dy, dz) in enumerate(faces):
                        if face_vis[x, y, z, n]:
                            normal_list.append([x + 0.5 * dx, y + 0.5 * dy, z + 0.5 * dz, dx, dy, dz])
    
    
    # Create Open3D voxel grid
    voxel_points = o3d.utility.Vector3dVector(voxel_list)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        o3d.geometry.PointCloud(points=voxel_points),
        voxel_size=1.0,
        min_bound=[0, 0, 0],
        max_bound=[X, Y, Z]
    )
    
    # Create Open3D lines for normals
    line_set = o3d.geometry.LineSet()
    line_points = []
    lines = []
    colors = []

    for i, (px, py, pz, nx, ny, nz) in enumerate(normal_list):
        line_points.append([px, py, pz])
        line_points.append([px + nx, py + ny, pz + nz])
        lines.append([2 * i, 2 * i + 1])
        colors.append([1, 0, 0])  # red color for normals

    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize using Open3D
    o3d.visualization.draw_geometries([voxel_grid, line_set])


def main():
    scenes_path = sorted(glob.glob(os.path.join(args_cli.input, '**', 'occ.npy'), recursive=True))
    #print(scenes_path)
    print(len(scenes_path))

    for scene_path in scenes_path:
        occ_set, z_len, x_len, y_len = fill_grid(scene_path)
        face_vis = save_face(scene_path, occ_set)
        #vis_face_and_voxel(scene_path, face_vis) 
        hollow_set = convert_to_hollow(occ_set, min_bound=[0, 0, 0], max_bound=[x_len, y_len, z_len])
        map_to_3d_grid(hollow_set, [0, 0, 0], [x_len, y_len, z_len], scene_path)
        #exit()
        #print(scene_path)

if __name__=="__main__":
    main()
