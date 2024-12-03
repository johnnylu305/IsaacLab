# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import os

import omni
import omni.kit.commands
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

from omni.isaac.lab.sim.converters.asset_converter_base import AssetConverterBase
from omni.isaac.lab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from omni.isaac.lab.sim.schemas import schemas
from omni.isaac.lab.sim.utils import export_prim_to_file
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.isaac.core.utils.prims as prims_utils
from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd
from pxr import Gf, Vt

import pxr
import numpy as np
import open3d as o3d
from PIL import Image



class MeshConverterRescaleMAD3D(AssetConverterBase):
    """Converter for a mesh file in OBJ / STL / FBX format to a USD file.

    This class wraps around the `omni.kit.asset_converter`_ extension to provide a lazy implementation
    for mesh to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    To make the asset instanceable, we must follow a certain structure dictated by how USD scene-graph
    instancing and physics work. The rigid body component must be added to each instance and not the
    referenced asset (i.e. the prototype prim itself). This is because the rigid body component defines
    properties that are specific to each instance and cannot be shared under the referenced asset. For
    more information, please check the `documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#instancing-rigid-bodies>`_.

    Due to the above, we follow the following structure:

    * ``{prim_path}`` - The root prim that is an Xform with the rigid body and mass APIs if configured.
    * ``{prim_path}/geometry`` - The prim that contains the mesh and optionally the materials if configured.
      If instancing is enabled, this prim will be an instanceable reference to the prototype prim.

    .. _omni.kit.asset_converter: https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html

    .. caution::
        When converting STL files, Z-up convention is assumed, even though this is not the default for many CAD
        export programs. Asset orientation convention can either be modified directly in the CAD program's export
        process or an offset can be added within the config in Isaac Lab.

    """

    cfg: MeshConverterCfg
    """The configuration instance for mesh to USD conversion."""

    def __init__(self, cfg: MeshConverterCfg, max_len: 8):
        """Initializes the class.

        Args:
            cfg: The configuration instance for mesh to USD conversion.
        """
        self.max_len = max_len
        super().__init__(cfg=cfg)


    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: MeshConverterCfg):
        """Generate USD from OBJ, STL or FBX.

        It stores the asset in the following format:

        /file_name (default prim)
          |- /geometry <- Made instanceable if requested
            |- /Looks
            |- /???

        Args:
            cfg: The configuration for conversion of mesh to USD.

        Raises:
            RuntimeError: If the conversion using the Omniverse asset converter fails.
        """
        # resolve mesh name and format
        mesh_file_basename, mesh_file_format = os.path.basename(cfg.asset_path).split(".")
        mesh_file_format = mesh_file_format.lower()
        
        mesh_file_basename = 'a' + mesh_file_basename

        # Convert USD
        asyncio.get_event_loop().run_until_complete(
            self._convert_mesh_to_usd(
                in_file=cfg.asset_path, out_file=self.usd_path, prim_path=f"/{mesh_file_basename}"
            )
        )
        # Open converted USD stage
        # note: This opens a new stage and does not use the stage created earlier by the user
        # create a new stage
        stage = Usd.Stage.Open(self.usd_path)
        #import pdb; pdb.set_trace()
        # add USD to stage cache
        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        # Get the default prim (which is the root prim) -- "/{mesh_file_basename}"
        xform_prim = stage.GetDefaultPrim()
        geom_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}/geometry")

        # Make sure the prim is an Xformable type (which can have transformations like scale)
        xformable = UsdGeom.Xformable(geom_prim)

        # Check if the prim has a scale transformation
        scale_attr = xformable.GetXformOpOrderAttr().Get()

        # # Clear the scale transform if it exists
        # for op in xformable.GetOrderedXformOps():
        #     if op.GetOpType() == UsdGeom.XformOp.TypeScale:
        #         # Remove the scale operation
        #         op.GetAttr().Clear()
        #         print(f"Scale cleared for prim: {geom_prim.GetPath()}")
        # Rescale
        
        #rescale_scene(scene_prim_root=f"/{mesh_file_basename}/geometry")

        rescale_scene(stage, self.usd_path, scene_prim_root=f"/{mesh_file_basename}/geometry", max_len=self.max_len)
        
        # Move all meshes to underneath new Xform
        stack = [geom_prim]
        # While there are nodes in the stack
        while stack:
            # Pop the last node from the stack
            node = stack.pop()
            if node.GetTypeName() == "Mesh":
                # Apply collider properties to mesh
                if cfg.collision_props is not None:
                    # -- Collision approximation to mesh
                    # TODO: https://github.com/isaac-orbit/orbit/issues/163 Move this to a new Schema
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(node)
                    mesh_collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
                    # -- Collider properties such as offset, scale, etc.
                    schemas.define_collision_properties(
                        prim_path=node.GetPath(), cfg=cfg.collision_props, stage=stage
                    )

            # Get the children of the current node
            children = node.GetChildren()
    
            # Iterate over each child
            for child in children:
                # Add the child to the stack for further traversal
                stack.append(child)

        # Delete the old Xform and make the new Xform the default prim
        stage.SetDefaultPrim(xform_prim)
        # Handle instanceable
        # Create a new Xform prim that will be the prototype prim
        if cfg.make_instanceable:
            # Export Xform to a file so we can reference it from all instances
            export_prim_to_file(
                path=os.path.join(self.usd_dir, self.usd_instanceable_meshes_path),
                source_prim_path=geom_prim.GetPath(),
                stage=stage,
            )
            # Delete the original prim that will now be a reference
            geom_prim_path = geom_prim.GetPath().pathString
            omni.kit.commands.execute("DeletePrims", paths=[geom_prim_path], stage=stage)
            # Update references to exported Xform and make it instanceable
            geom_undef_prim = stage.DefinePrim(geom_prim_path)
            geom_undef_prim.GetReferences().AddReference(self.usd_instanceable_meshes_path, primPath=geom_prim_path)
            geom_undef_prim.SetInstanceable(True)

        # Apply mass and rigid body properties after everything else
        # Properties are applied to the top level prim to avoid the case where all instances of this
        #   asset unintentionally share the same rigid body properties
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path=xform_prim.GetPath(), cfg=cfg.mass_props, stage=stage)
        # apply rigid body properties
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(prim_path=xform_prim.GetPath(), cfg=cfg.rigid_props, stage=stage)
        
        # Save changes to USD stage
        
        stage.Save()
        if stage_id is not None:
            UsdUtils.StageCache.Get().Erase(stage_id)

    """
    Helper methods.
    """

    @staticmethod
    async def _convert_mesh_to_usd(
        in_file: str, out_file: str, prim_path: str = "/World", load_materials: bool = True
    ) -> bool:
        """Convert mesh from supported file types to USD.

        This function uses the Omniverse Asset Converter extension to convert a mesh file to USD.
        It is an asynchronous function and should be called using `asyncio.get_event_loop().run_until_complete()`.

        The converted asset is stored in the USD format in the specified output file.
        The USD file has Y-up axis and is scaled to meters.

        The asset hierarchy is arranged as follows:

        .. code-block:: none
            prim_path (default prim)
                |- /geometry/Looks
                |- /geometry/mesh

        Args:
            in_file: The file to convert.
            out_file: The path to store the output file.
            prim_path: The prim path of the mesh.
            load_materials: Set to True to enable attaching materials defined in the input file
                to the generated USD mesh. Defaults to True.

        Returns:
            True if the conversion succeeds.
        """
        #prim_path = '/a'+prim_path[1:]
        enable_extension("omni.kit.asset_converter")
        enable_extension("omni.usd.metrics.assembler")

        import omni.kit.asset_converter
        import omni.usd
        from omni.metrics.assembler.core import get_metrics_assembler_interface

        # Create converter context
        converter_context = omni.kit.asset_converter.AssetConverterContext()
        # Set up converter settings
        # Don't import/export materials
        converter_context.ignore_materials = not load_materials
        converter_context.ignore_animations = True
        converter_context.ignore_camera = True
        converter_context.ignore_light = True
        # Merge all meshes into one
        converter_context.merge_all_meshes = True #True
        # Sets world units to meters, this will also scale asset if it's centimeters model.
        # This does not work right now :(, so we need to scale the mesh manually
        converter_context.use_meter_as_world_unit = True
        converter_context.baking_scales = True
        # Uses double precision for all transform ops.
        converter_context.use_double_precision_to_usd_transform_op = True

        # Create converter task
        instance = omni.kit.asset_converter.get_instance()
        out_file_non_metric = out_file.replace(".usd", "_non_metric.usd")
        task = instance.create_converter_task(in_file, out_file_non_metric, None, converter_context)
        # Start conversion task and wait for it to finish
        success = True
        while True:
            success = await task.wait_until_finished()
            if not success:
                await asyncio.sleep(0.1)
            else:
                break

        temp_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(temp_stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(temp_stage, 1.0)
        #import pdb; pdb.set_trace()
        base_prim = temp_stage.DefinePrim(prim_path, "Xform")
        prim = temp_stage.DefinePrim(f"{prim_path}/geometry", "Xform")
        prim.GetReferences().AddReference(out_file_non_metric)
        cache = UsdUtils.StageCache.Get()
        cache.Insert(temp_stage)
        stage_id = cache.GetId(temp_stage).ToLongInt()
        get_metrics_assembler_interface().resolve_stage(stage_id)
        temp_stage.SetDefaultPrim(base_prim)
        temp_stage.Export(out_file)
        return success

def get_all_mesh_prim_path(stage, root):
    root_prim = stage.GetPrimAtPath(root)
    #import pdb; pdb.set_trace()
    stack = [root_prim]
    mesh_prim_path = []
    # While there are nodes in the stack
    while stack:
        # Pop the last node from the stack
        node = stack.pop()
        if node.GetTypeName() == "Mesh":
            mesh_prim_path.append(node.GetPath().pathString)
        # Get the children of the current node
        children = node.GetChildren()
    
        # Iterate over each child
        for child in children:
            # Add the child to the stack for further traversal
            stack.append(child)
    return mesh_prim_path


'''
def get_minmax_mesh_coordinates(mesh_prim):
    # Access the mesh's point positions in local space
    mesh = UsdGeom.Mesh(mesh_prim)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()

    # Get the world transformation matrix for the mesh
    xformable = UsdGeom.Xformable(mesh_prim)
    #world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # Transform each point to world coordinates
    #transformed_points = [world_transform.Transform(point) for point in points]
    transformed_points = points

    # Calculate the maximum coordinates
    max_coords = Gf.Vec3f(float('-inf'), float('-inf'), float('-inf'))
    min_coords = Gf.Vec3f(float('inf'), float('inf'), float('inf'))
    for point in transformed_points:
        max_coords[0] = max(max_coords[0], point[0])
        max_coords[1] = max(max_coords[1], point[1])
        max_coords[2] = max(max_coords[2], point[2])

        min_coords[0] = min(min_coords[0], point[0])
        min_coords[1] = min(min_coords[1], point[1])
        min_coords[2] = min(min_coords[2], point[2])
    print(max_coords, min_coords)
    return max_coords, min_coords
'''


def get_minmax_mesh_coordinates(mesh_prim):
    # Access the mesh's point positions in local space
    mesh = UsdGeom.Mesh(mesh_prim)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()

    # Get the world transformation matrix for the mesh
    xformable = UsdGeom.Xformable(mesh_prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # Transform each point to world coordinates
    transformed_points = [world_transform.Transform(point) for point in points]

    # Calculate the maximum coordinates
    max_coords = Gf.Vec3f(float('-inf'), float('-inf'), float('-inf'))
    min_coords = Gf.Vec3f(float('inf'), float('inf'), float('inf'))
    for point in transformed_points:
        max_coords[0] = max(max_coords[0], point[0])
        max_coords[1] = max(max_coords[1], point[1])
        max_coords[2] = max(max_coords[2], point[2])

        min_coords[0] = min(min_coords[0], point[0])
        min_coords[1] = min(min_coords[1], point[1])
        min_coords[2] = min(min_coords[2], point[2])

    return max_coords, min_coords

def get_centroid(stage, mesh_prim_path):
    # Access the mesh's point positions in local space
    
    num_points=0
    centroid = Gf.Vec3f(0.0, 0.0, 0.0)
    min_z = float('inf')
    min_y = float('inf')
    min_x = float('inf')
    max_x, max_y, max_z = -1e10, -1e10, -1e10
    
    
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    # Compute the world space bounding box for the specified mesh primitive
    bbox = bbox_cache.ComputeWorldBound(mesh_prim)
    min_coords = bbox.GetBox().GetMin()
    max_coords = bbox.GetBox().GetMax()
    
    '''
    for prim_path in mesh_prim_path:
        mesh_prim = stage.GetPrimAtPath(prim_path)
        max_coords, min_coords = get_minmax_mesh_coordinates(mesh_prim)
        
        mesh = UsdGeom.Mesh(mesh_prim)
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get()
    
        # Get the world transformation matrix for the mesh
        xformable = UsdGeom.Xformable(mesh_prim)
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        #import pdb; pdb.set_trace()
        # Transform each point to world coordinates
        transformed_points = [world_transform.Transform(point) for point in points]
    
        # Initialize min/max coordinates and centroid
        max_coords = Gf.Vec3f(float('-inf'), float('-inf'), float('-inf'))
        min_coords = Gf.Vec3f(float('inf'), float('inf'), float('inf'))
        
    
        # Calculate min/max coordinates and accumulate for centroid
        num_points += len(transformed_points)
        for point in transformed_points:
            max_coords[0] = max(max_coords[0], point[0])
            max_coords[1] = max(max_coords[1], point[1])
            max_coords[2] = max(max_coords[2], point[2])
    
            min_coords[0] = min(min_coords[0], point[0])
            min_coords[1] = min(min_coords[1], point[1])
            min_coords[2] = min(min_coords[2], point[2])
            
            min_z = min(min_z, min_coords[2])
            min_y = min(min_y, min_coords[1])
            min_x = min(min_x, min_coords[0])
            max_x = max(max_x, max_coords[0])
            max_y = max(max_y, max_coords[1])
            max_z = max(max_z, max_coords[2])
    
            # Add the point to the centroid accumulator
            centroid += point
    
    # Calculate the centroid by dividing the sum of points by the number of points
    if num_points > 0:
        centroid /= num_points
    '''
    #centroid[2] = (min_z+max_z)/2
    #centroid[1] = min_y
    #centroid[0] = (min_x+max_x)/2
    
    centroid[2] = (min_coords[2]+ max_coords[0])/2
    centroid[1] = min_coords[1]
    centroid[0] = (min_coords[0] + max_coords[0])/2
    return centroid

def get_scale(stage, mesh_prim_path, desired_len):
    
    max_x, max_y, max_z = -1e10, -1e10, -1e10
    min_x, min_y, min_z = 1e10, 1e10, 1e10

    for prim_path in mesh_prim_path:
        mesh_prim = stage.GetPrimAtPath(prim_path)
        max_coords, min_coords = get_minmax_mesh_coordinates(mesh_prim)
   
        #print(max_coords, min_coords)

        max_x = max(max_x, max_coords[0])
        max_y = max(max_y, max_coords[1])
        max_z = max(max_z, max_coords[2])
        min_x = min(min_x, min_coords[0])
        min_y = min(min_y, min_coords[1])
        min_z = min(min_z, min_coords[2])
    extent = (max_x-min_x, max_y-min_y, max_z-min_z)
    max_side = max(extent)
    print(f"Max Side: {max_side} meters")
    return desired_len/max_side

def convert_to_open3d_mesh(mesh_data):
    vertices, face_vertex_counts, face_vertex_indices = mesh_data
    vertices = np.array(vertices)
    triangles = []

    # Convert face indices
    index = 0
    for fvc in face_vertex_counts:
        if fvc == 3:  # Assuming all faces are triangles
            triangles.append(face_vertex_indices[index:index+3])
        index += fvc

    triangles = np.array(triangles)

    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.compute_vertex_normals()  # Optional: for better visualization
    return o3d_mesh

def rescale_mesh(mesh, scale_factor):
    vertices = np.asarray(mesh.vertices)
    centroid = mesh.get_center()
    
    centroid[1] = vertices[:,1].min()
    vertices -= centroid
    vertices *= scale_factor  # Scale the vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()  # Recompute normals for proper visualization
    return mesh
    
def extract_mesh_data(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return vertices, faces
    
def create_usd_from_mesh(vertices, faces, usd_file_path):
    # Create a new stage
    stage = Usd.Stage.CreateNew(usd_file_path)

    # Create a mesh primitive at the specified path
    mesh = UsdGeom.Mesh.Define(stage, '/World/myMesh')

    # Set vertices positions
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(vertices.tolist()))

    # Set face vertex indices
    face_vertex_counts = [len(face) for face in faces]
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))

    # Flatten the list of faces to a list of indices
    face_vertex_indices = faces.flatten()
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices.tolist()))

    # Save the stage
    stage.Save()

    return stage

def voxelize_mesh(mesh, voxel_size):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    return voxel_grid

def initialize_occupancy_grid(x_size, y_size, z_size):
    # Define the grid size and resolution (cell size)
    grid = np.zeros((x_size, y_size, z_size))
    return grid

def save_occupancy_grid_as_image(occupancy_grid, filename):
    # Define the color mapping for the occupancy grid
    colors = {
        1: (0, 0, 0),       # Black for occupied
        0: (255, 255, 255), # White for unoccupied
        2: (255, 255, 255)  # Assume it is free
    }

    # Create an RGB image from the occupancy grid
    image_data = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1], 3), dtype=np.uint8)
    for value, color in colors.items():
        image_data[occupancy_grid == value] = color

    # Create and save the image
    image = Image.fromarray(image_data, 'RGB')
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(np.array(image).shape)
    image.save(filename)
    print(f"Saved occupancy map as image: {filename}")

def rescale_scene(stage,outfile, scene_prim_root="/World/Scene",max_len=8):
    all_points = []
    meshes = []
    #import pdb; pdb.set_trace()
    mesh_prim_path = get_all_mesh_prim_path(stage,scene_prim_root)
    
    for prim_path in mesh_prim_path:
        #mesh_prim = get_prim_at_path(prim_path)
        mesh_prim = stage.GetPrimAtPath(prim_path)
        mesh = UsdGeom.Mesh(mesh_prim)
        #import pdb; pdb.set_trace()
        
        points = mesh.GetPointsAttr().Get()
        if points:
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            meshes.append((points, face_vertex_counts, face_vertex_indices))
            
            points_attr = mesh.GetPointsAttr()
            points = points_attr.Get()
            xformable = UsdGeom.Xformable(mesh_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
            # Transform each point to world coordinates
            transformed_points = [world_transform.Transform(point) for point in points]
            if transformed_points:
                all_points.extend(transformed_points)  # Aggregate points
    

    mesh_prim_path = get_all_mesh_prim_path(stage,scene_prim_root)
    # for prim_path in mesh_prim_path:
    #     mesh_prim = get_prim_at_path(prim_path=prim_path)
    #     clear_transforms_for_parents(mesh_prim)

    # Get the root prim of the scene
    root_prim = stage.GetPrimAtPath(scene_prim_root)

    # Initialize cumulative rotation as an identity quaternion
    cumulative_rotation = Gf.Quatf(1, 0, 0, 0)  # Identity quaternion
    original_up_vector = Gf.Quatf(0, 0, 0, 1)
    # Traverse from root prim to the last child and accumulate rotations

    current_prim = stage.GetPrimAtPath(mesh_prim_path[0])
    while current_prim != root_prim:
        # Check if the current prim has a transform
        if current_prim.IsA(UsdGeom.Xform):
            xform = UsdGeom.Xform(current_prim)
            xform_ops = xform.GetOrderedXformOps()

            # Extract and accumulate rotation quaternions
            for op in xform_ops:
                print(op.GetOpType())
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    rotation_quaternion = op.Get()  # Get quaternion as Gf.Quatf or Gf.Quatd
                    print(rotation_quaternion)
                    cumulative_rotation = rotation_quaternion * cumulative_rotation
        # Move to the parent prim
        current_prim = current_prim.GetParent()
        
    # Output the final cumulative rotation
    print("Cumulative Rotation Quaternion from Root to Last Child:", cumulative_rotation)
    # Convert the up vector into a quaternion with zero scalar part
    #v_quat = Gf.Quatd(0, original_up_vector)

    # Apply the quaternion rotation: rotated_vector = q * v * q.inverse()
    rotated_quaternion = cumulative_rotation * original_up_vector * cumulative_rotation.GetInverse()

    # Extract the vector part of the resulting quaternion
    up_axis = rotated_quaternion.GetImaginary()
    print("Transformed Up Axis:", up_axis)
    abs_up_axis = Gf.Vec3d(abs(up_axis[0]), abs(up_axis[1]), abs(up_axis[2]))
    if abs_up_axis[0] >= abs_up_axis[1] and abs_up_axis[0] >= abs_up_axis[2]:
        # X-axis is dominant
        if up_axis[0] > 0:
            mapped_axis = "x"
        else:
            mapped_axis = "-x"
    elif abs_up_axis[1] >= abs_up_axis[0] and abs_up_axis[1] >= abs_up_axis[2]:
        # Y-axis is dominant
        if up_axis[1] > 0:
            mapped_axis = "y"
        else:
            mapped_axis = "-y"
    else:
        # Z-axis is dominant
        if up_axis[2] > 0:
            mapped_axis = "z"
        else:
            mapped_axis = "-z"

    print("Mapped Up Axis:", mapped_axis)
    
        


    centers=[]
    y_mins=[0]
    x_mins=[0]
    z_mins=[0]

    y_maxs=[0]
    x_maxs=[0]
    z_maxs=[0]
    voxel_grids = []
    
    for mesh_data in meshes:
        o3d_mesh = convert_to_open3d_mesh(mesh_data)

        # Compute minimal y-coordinate
        y_min = o3d_mesh.get_min_bound()[1]
        x_min = o3d_mesh.get_min_bound()[0]
        z_min = o3d_mesh.get_min_bound()[2]

        y_max = o3d_mesh.get_max_bound()[1]
        x_max = o3d_mesh.get_max_bound()[0]
        z_max = o3d_mesh.get_max_bound()[2]

        center = o3d_mesh.get_center()
       
        o3d_mesh.translate(-center)  # Center the mesh by translating it to the origin based on the adjusted center
        print("Adjusted Center:", center)

        # scale_factor = args_cli.max_len / max(o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound())
        # o3d_mesh.scale(scale_factor, center=(0, 0, 0))
        

        # Compute the center and adjust the y-coordinate to the minimum y

        # Store the adjusted center and y_min for later calculations
        centers.append(center)
        y_mins.append(y_min)
        x_mins.append(x_min)
        z_mins.append(z_min)

        y_maxs.append(y_max)
        x_maxs.append(x_max)
        z_maxs.append(z_max)


    #import pdb; pdb.set_trace()
    # Calculate the mean center
    mean_center = np.mean(centers, axis=0)
    global_y_min = min(y_mins)
    global_x_min = min(x_mins)
    global_z_min = min(z_mins)

    global_y_max = max(y_maxs)
    global_x_max = max(x_maxs)
    global_z_max = max(z_maxs)
    mean_center[0] = (global_x_min+global_x_max)/2 
    mean_center[1] = (global_y_min+global_y_max)/2 
    mean_center[2] = (global_z_min+global_z_max)/2 
    # if mapped_axis == '-y':
    #     mean_center[2] = global_z_min
    # if mapped_axis == '-x':
    #     mean_center[0] = global_x_min
    # if mapped_axis == '-z':
    #     mean_center[1] = global_y_max
    # if mapped_axis == 'y':
    #     mean_center[2] = global_z_max
    # if mapped_axis == 'x':
    #     mean_center[0] = global_x_max
    # if mapped_axis == 'z':
    #     mean_center[1] = global_y_min
    centroid = mean_center

    mesh_prim_path = get_all_mesh_prim_path(stage,scene_prim_root)
    # for prim_path in mesh_prim_path:
    #     mesh_prim = get_prim_at_path(prim_path=prim_path)
    #     clear_transforms_for_parents(mesh_prim)

    root_prim = stage.GetPrimAtPath(scene_prim_root)

    # Traverse all descendants of the root prim
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Xform):  # Check if the prim has transform attributes
            xform = UsdGeom.Xform(prim)
            #Get all transformation operations on the prim
            xform_ops = xform.GetOrderedXformOps()

            # Iterate through each operation and clear translation and scale transformations
            for op in xform_ops:
                op_type = op.GetOpType()
                
                # Clear only translation and scale types
                #if op_type in (UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeScale):
                #    op.GetAttr().Clear()  # Clear the specific transformation operation
                if op_type == UsdGeom.XformOp.TypeScale:
                    op.GetAttr().Clear()  # Clear the specific transformation operation


            # Update xformOpOrder to remove cleared operations, keeping only rotations
            new_xform_op_order = [op for op in xform.GetOrderedXformOps() if op.GetOpType() not in (UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeScale)]
            xform.SetXformOpOrder(new_xform_op_order)

            # Clear the transformation by setting identity transforms
            #xform.ClearXformOpOrder()  # Clears all transform operations
            #xform.AddTransformOp().Set(Gf.Matrix4d(1.0))  # Sets identity matrix as the transform

    print(mesh_prim_path)

    scale_factor = get_scale(stage, mesh_prim_path, max_len)
    

    mesh_prim_path = get_all_mesh_prim_path(stage,scene_prim_root)
    merged_points = []
    merged_face_vertex_indices = []
    merged_face_vertex_counts = []
    vertex_offset = 0
    for prim_path in mesh_prim_path:
        # Get the mesh prim
        mesh_prim = stage.GetPrimAtPath(prim_path)
        mesh = UsdGeom.Mesh(mesh_prim)               
        
        # Get points (vertices), face vertex indices, and face vertex counts
        points = mesh.GetPointsAttr().Get()  # List of Gf.Vec3f or Gf.Vec3d
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()  # Indices list
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()  # List of face sizes (triangles/quads)

        # Add points to merged points list
        merged_points.extend(points)
        
        # Offset and add face vertex indices to merged list
        adjusted_indices = [i + vertex_offset for i in face_vertex_indices]
        merged_face_vertex_indices.extend(adjusted_indices)
        
        # Add face counts directly (no need for offset here)
        merged_face_vertex_counts.extend(face_vertex_counts)
        
        # Update vertex offset
        vertex_offset += len(points)

    # Create a single tuple with merged points, face counts, and face indices
    merged_mesh = (merged_points, merged_face_vertex_counts, merged_face_vertex_indices)

    #glb_file_path = "/home/hat/Documents/Dataset/objaverse/hf-objaverse-v1/glbs/000-000/000a00944e294f7a94f95d420fdd45eb.glb"

    # Load the mesh from the .glb file


    o3d_mesh = convert_to_open3d_mesh(merged_mesh)

    #o3d_mesh = o3d.io.read_triangle_mesh(glb_file_path)
    # # Rescale the mesh
    scale_factor = max_len / max(o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound())
    print(scale_factor)
    print(f"Scaling factor: {scale_factor}")
    #import pdb; pdb.set_trace()
    #o3d_mesh.translate((-centroid[0], -centroid[1], -centroid[2]))
    shift_xyz = o3d_mesh.get_center()
    o3d_mesh.translate(-shift_xyz)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([np.radians(90), 0, 0])
    o3d_mesh.rotate(rotation_matrix, center=(0, 0, 0))
    o3d_mesh.scale(scale_factor, center=(0, 0, 0))

    z_min = o3d_mesh.get_min_bound()[2]
    o3d_mesh.translate((0,0,-z_min))
    scaled_points = Vt.Vec3fArray([tuple(vertex) for vertex in np.asarray(o3d_mesh.vertices)])
    offset=0

    #o3d.visualization.draw([o3d_mesh])
    voxel_size=0.5
    voxel_grid = voxelize_mesh(o3d_mesh, voxel_size)
    #o3d.visualization.draw([voxel_grid])
    voxel_coordinates = np.array([voxel_grid.origin + voxel.grid_index * voxel_size for voxel in voxel_grid.get_voxels()])
    voxel_centers = np.round(voxel_coordinates).astype(int)
    #import pdb; pdb.set_trace()
    occ_grid = initialize_occupancy_grid(20,20,20)
    for i in range(len(voxel_centers)):
        occ_grid[voxel_centers[i][0]+10,voxel_centers[i][1]+10,voxel_centers[i][2]]=1
    
    folder = os.path.dirname(outfile)
    for i in range(occ_grid.shape[2]):
        # vis occ
        image_filename = f"occupancy_map_slice_{i}.png"
        save_occupancy_grid_as_image(occ_grid[:, :, i], os.path.join(folder, image_filename))
        # Save as npy file
        np.save(os.path.join(folder, "occ.npy"), occ_grid[:, :, :])

    
    for prim_path in mesh_prim_path:
        mesh_prim = stage.GetPrimAtPath(prim_path)
        
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()  # List of Gf.Vec3f or Gf.Vec3d
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()  # Indices list
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()  # List of face sizes (triangles/quads)

        # o3d_mesh = convert_to_open3d_mesh((points,face_vertex_counts, face_vertex_indices))
        # o3d_mesh.translate((-centroid[0], -centroid[1], -centroid[2]))
        # o3d_mesh.scale(scale_factor, center=(0, 0, 0))
        
        # Convert back the scaled Open3D mesh to USD-compatible format
        
        len_points = len(mesh.GetPointsAttr().Get())
        
        # Assign the scaled points back to the original USD mesh prim
        #import pdb; pdb.set_trace()
        mesh.GetPointsAttr().Set(scaled_points[offset:offset+len_points])
        offset+=len_points
        #mesh.GetPointsAttr().Set(scaled_points)

    
    # mesh_prim = stage.GetPrimAtPath(scene_prim_root)
    
    # #clear_transforms_for_parents(mesh_prim)
    # xform = UsdGeom.Xformable(mesh_prim)
    # xform.ClearXformOpOrder()
    # # Define the rotation matrix for -90 degrees about the X-axis
    # #rotation_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90))

    # # Get all transformation operations on the prim
    # #xform_ops = xform.GetOrderedXformOps()

    # shift_transform = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-shift_xyz))
    # scale_transform = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
    # #xform.ClearXformOpOrder()  # Clear any existing transformations
    # #xform.AddTransformOp().Set(scale_transform)
    
    # combined_transform = shift_transform*scale_transform
    # #combined_transform = scale_transform
    # xform.AddTransformOp().Set(combined_transform)
    # stage.Save()

    #original_prim_path = "/World/Scene/geometry"

    # Check if the original prim exists, and remove it if it does
    #original_prim = stage.GetPrimAtPath(original_prim_path)

    #stage.RemovePrim(original_prim_path)



    

    


