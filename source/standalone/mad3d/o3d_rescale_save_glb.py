import open3d as o3d
import numpy as np
import glob
import os

# Input and output paths
input_path = "/home/hat/Documents/Dataset/objaverse/hf-objaverse-v1/glbs"  # Replace with your input folder
output_path = "/home/hat/Documents/Dataset/objaverse_rescaled_glb/"  # Replace with your output folder

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Get all .glb files in the input directory recursively
glb_file_paths = sorted(glob.glob(os.path.join(input_path, '**', '*.glb'), recursive=True))

# Maximum desired length for the rescaled mesh
max_len = 8.0  # Replace with your desired maximum size


from pygltflib import GLTF2
import numpy as np
import open3d as o3d

# # Load the GLTF file
# gltf_file = "path/to/your_file.glb"

# Process each .glb file
for glb_file_path in glb_file_paths:
    try:

        gltf = GLTF2().load(glb_file_path)

        # Assume the first node contains the transformation matrix (modify as needed)
        node = gltf.nodes[0]  # Replace with the correct node index if necessary

        # Extract the transformation matrix (identity if none exists)
        if node.matrix:
            transform_matrix = np.array(node.matrix).reshape(4, 4)
        else:
            transform_matrix = np.eye(4)  # Default to identity matrix

        print("Original Transformation Matrix:")
        print(transform_matrix)

        # Load the mesh from the .glb file
        o3d_mesh = o3d.io.read_triangle_mesh(glb_file_path)

        # Check if the mesh contains textures
        if not o3d_mesh.textures:
            print(f"Warning: No textures found in {glb_file_path}")
            continue

        print(f"Processing {glb_file_path}")

        # Rescale the mesh
        scale_factor = max_len / max(o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound())
        print(f"Scaling factor: {scale_factor}")

        # Rotate to match the desired axis
        #rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([np.radians(90), 0, 0])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([np.radians(180), 0, 0])
        o3d_mesh.rotate(rotation_matrix, center=(0, 0, 0))

        # Apply the original transformation matrix
        R = transform_matrix[:3, :3]  # Rotation part
        #t = transform_matrix[:3, 3]   # Translation part

        # Rotate and translate the Open3D mesh
        o3d_mesh.rotate(R, center=(0, 0, 0))
        #o3d_mesh.translate(t)

        # Translate to center
        shift_xyz = o3d_mesh.get_center()
        o3d_mesh.translate(-shift_xyz)
        # Scale the mesh
        o3d_mesh.scale(scale_factor, center=(0, 0, 0))

        # Translate the mesh to set the minimum Z coordinate to 0
        z_max = o3d_mesh.get_max_bound()[2]
        o3d_mesh.translate((0, 0, -z_max))
        
        # Output file path
        output_obj_path = os.path.join(output_path, os.path.splitext(os.path.basename(glb_file_path))[0] + ".obj")

        # Save the transformed mesh with texture as .obj
        o3d.io.write_triangle_mesh(output_obj_path, o3d_mesh, write_triangle_uvs=True)
        print(f"Mesh with texture saved as {output_obj_path}")

    except Exception as e:
        print(f"Error processing {glb_file_path}: {e}")
