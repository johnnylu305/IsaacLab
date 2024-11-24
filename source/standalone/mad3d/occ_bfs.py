import numpy as np
from collections import deque
import os

import argparse
import glob
import pickle

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to create occupancy grid")
parser.add_argument("input", type=str, help="The root to the input USD file.")
parser.add_argument("output", type=str, help="The root to store the occupancy grid.")
args_cli = parser.parse_args()
# Define the 3D occupancy grid (replace this with the actual grid)
occ_grid = np.zeros((20, 20, 20), dtype=int)  # Example 3D grid of 20x20x20
# Set some occupied voxels as an example (use 1 for occupied voxels)
occ_grid[5:15, 5:15, 5:15] = 1

# Start position (outside any occupied voxel)
start_position = (19, 19, 19)

# Check that start position is empty
if occ_grid[start_position] != 0:
    raise ValueError("Start position must be an empty voxel.")

# BFS to find the outermost layer of occupied voxels
def get_outermost_layer(occ_grid, start_position):
    # Define 6 possible neighbors in 3D (up, down, left, right, front, back)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    visited = np.zeros_like(occ_grid, dtype=bool)
    outermost_layer = set()
    
    # Initialize BFS queue
    queue = deque([start_position])
    visited[start_position] = True

    while queue:
        x, y, z = queue.popleft()
        
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            # Check if the neighbor is within bounds
            if 0 <= nx < occ_grid.shape[0] and 0 <= ny < occ_grid.shape[1] and 0 <= nz < occ_grid.shape[2]:
                # If the neighbor is occupied and not visited, check if it's on the outermost layer
                if occ_grid[nx, ny, nz] == 1:
                    # Check if this occupied voxel has at least one empty neighbor (i.e., is on the outermost layer)
                    is_outermost = any(
                        0 <= nx + ddx < occ_grid.shape[0] and
                        0 <= ny + ddy < occ_grid.shape[1] and
                        0 <= nz + ddz < occ_grid.shape[2] and
                        occ_grid[nx + ddx, ny + ddy, nz + ddz] == 0
                        for ddx, ddy, ddz in directions
                    )
                    if is_outermost:
                        outermost_layer.add((nx, ny, nz))
                # If neighbor is empty and not visited, add to queue for further exploration
                elif occ_grid[nx, ny, nz] == 0 and not visited[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    queue.append((nx, ny, nz))

    return outermost_layer

#file_path = r'/home/hat/Documents/occ_objaverse/glbs/000-000/0a6ad1a88cc04756ba4743badb098f90/'

file_path = args_cli.input

occ_paths = sorted(glob.glob(os.path.join(args_cli.input, '**', '*.npy'), recursive=True))

for occ_path in occ_paths:
    occ_grid = np.load(occ_path)

    occ_grid[:,:,0] = 1 # add the floor for bfs
    # Get the outermost layer of occupied voxels
    outermost_layer = get_outermost_layer(occ_grid, start_position)
    print("Outermost Layer of Occupied Voxels:", outermost_layer)

    # Initialize a new empty occupancy grid with the same shape
    modified_occ_grid = np.zeros(occ_grid.shape, dtype=int)

    # Mark the outermost layer in the new grid
    for coord in outermost_layer:
        modified_occ_grid[coord] = 1  # Mark voxel as occupied in the outermost layer

    relative_path = os.path.relpath(occ_path, args_cli.input)
    dest_path = os.path.join(args_cli.output, relative_path)
    output = os.path.split(dest_path)[0]

    #output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output, exist_ok=True)
    # Save as a .npy file
    np.save(os.path.join(output,"modified_occ_grid.npy"), modified_occ_grid)
    # Save the set as a binary file using pickle
    with open(os.path.join(output,"occ_bfs.pkl"), "wb") as f:
        pickle.dump(outermost_layer, f)
    #import pdb; pdb.set_trace()




