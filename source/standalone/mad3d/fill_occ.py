import os
import glob
import numpy as np
import argparse
import pickle
from collections import deque



# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to fill occupancy grid")
parser.add_argument("input", type=str, help="The root to the input occupancy grid.")
# parse the arguments
args_cli = parser.parse_args()


def add_neighbors(occupied_voxels, grid_shape):
    # Directions for 6-connected neighbors in a 3D grid
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
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
    z_dim, x_dim, y_dim = grid.shape
    
    # Directions for 6-connected neighbors in a 3D grid
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    
    # BFS to mark all reachable empty voxels from the exterior
    def bfs(start):
        queue = deque([start])
        visited = set()
        visited.add(start)
        
        while queue:
            z, x, y = queue.popleft()
            for dz, dx, dy in directions:
                nz, nx, ny = z + dz, x + dx, y + dy
                if 0 <= nx < x_dim and 0 <= ny < y_dim and 0 <= nz < z_dim:
                    if grid[nz, nx, ny] == 0 and (nz, nx, ny) not in visited:
                        visited.add((nz, nx, ny))
                        queue.append((nz, nx, ny))
        
        return visited
    
    # Start BFS from an exterior voxel
    exterior_empty_voxels = bfs((z_dim-1, x_dim-1, y_dim-1))
    
    occupied_voxels = set()
    
    # Collect all occupied voxels and mark all non-reachable empty voxels as occupied
    for z in range(z_dim):
        for x in range(x_dim):
            for y in range(y_dim):
                if grid[z, x, y] == 1 or (grid[z, x, y] == 0 and (z, x, y) not in exterior_empty_voxels):
                    occupied_voxels.add((z, x, y))
    
    return occupied_voxels

def fill_grid(path):
    hollow_occ = np.load(path)
    #print(hollow_occ.shape, np.sum(hollow_occ), hollow_occ[0, 0, 0])
    x_len, y_len, z_len = hollow_occ.shape
    assert hollow_occ[-1, -1, -1] == 0
    occ_set = find_occupied_voxels(hollow_occ)
    print(np.sum(hollow_occ), len(occ_set))
    #occ_set = add_neighbors(occ_set, hollow_occ.shape)
    #print(len(occ_set))
    output = os.path.join(os.path.split(path)[0], "fill_occ_set.pkl")
    # Save the occupied voxels to a file
    with open(output, 'wb') as file:
        pickle.dump(occ_set, file)
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
        if (vz, vx, vy) not in occ_grid:
            return False
    
    return True

def convert_to_hollow(occ_grid, min_bound, max_bound):
    hollowed_grid = set()
    
    for voxel in occ_grid:
        z, x, y = voxel
        if not is_hollow(x, y, z, occ_grid, min_bound, max_bound):
            hollowed_grid.add(voxel)
    
    return hollowed_grid

def map_to_3d_grid(hollow_set, min_bound, max_bound, path):
    # Calculate the dimensions of the grid
    x_dim = max_bound[0] - min_bound[0] 
    y_dim = max_bound[1] - min_bound[1] 
    z_dim = max_bound[2] - min_bound[2] 
    
    # Initialize the grid with zeros
    grid = np.zeros((z_dim, x_dim, y_dim), dtype=int)
    
    # Map the hollow set to the grid
    for (z, x, y) in hollow_set:
        grid[z - min_bound[0], x - min_bound[1], y - min_bound[2]] = 1

    output = os.path.join(os.path.split(path)[0], "hollow_occ.npy")
    np.save(output, grid)
    print("hollow", np.sum(grid))
    return grid

def main():
    scenes_path = sorted(glob.glob(os.path.join(args_cli.input, '**', 'occ.npy'), recursive=True))
    #print(scenes_path)
    print(len(scenes_path))

    for scene_path in scenes_path:
        occ_set, z_len, x_len, y_len = fill_grid(scene_path)
        hollow_set = convert_to_hollow(occ_set, min_bound=[0, 0, 0], max_bound=[x_len, y_len, z_len])
        map_to_3d_grid(hollow_set, [0, 0, 0], [x_len, y_len, z_len], scene_path)
        #exit()
        #print(scene_path)

if __name__=="__main__":
    main()
