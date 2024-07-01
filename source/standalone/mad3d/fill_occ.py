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
    for x in range(z_dim):
        for y in range(x_dim):
            for z in range(y_dim):
                if grid[z, x, y] == 1 or (grid[z, x, y] == 0 and (z, x, y) not in exterior_empty_voxels):
                    occupied_voxels.add((z, x, y))
    
    return occupied_voxels


def fill_grid(path):
    hollow_occ = np.load(path)
    #print(hollow_occ.shape, np.sum(hollow_occ), hollow_occ[0, 0, 0])
    #x_len, y_len, z_len = hollow_occ.shape
    assert hollow_occ[-1, -1, -1] == 0
    occ_set = find_occupied_voxels(hollow_occ)
    print(np.sum(hollow_occ), len(occ_set))
    #occ_set = add_neighbors(occ_set, hollow_occ.shape)
    #print(len(occ_set))
    output = os.path.join(os.path.split(path)[0], "fill_occ_set.pkl")
    # Save the occupied voxels to a file
    with open(output, 'wb') as file:
        pickle.dump(occ_set, file)


def main():
    scenes_path = sorted(glob.glob(os.path.join(args_cli.input, '**', 'occ.npy'), recursive=True))
    #print(scenes_path)
    print(len(scenes_path))

    for scene_path in scenes_path:
        fill_grid(scene_path)
        #print(scene_path)

if __name__=="__main__":
    main()
