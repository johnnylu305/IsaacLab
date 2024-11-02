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

    def __init__(self, cfg: MeshConverterCfg, max_len: 1):
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
        # add USD to stage cache
        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        # Get the default prim (which is the root prim) -- "/{mesh_file_basename}"
        xform_prim = stage.GetDefaultPrim()
        geom_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}/geometry")

        # Rescale
        rescale_scene(stage=stage, scene_prim_root=f"/{mesh_file_basename}/geometry", max_len=self.max_len)
        
        '''
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
        '''
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
        converter_context.merge_all_meshes = False #True
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
    # Create a bounding box cache with the specific time code
    #time_code = Usd.TimeCode(1)  # Adjust this if you need a different frame
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    # Compute the world space bounding box for the specified mesh primitive
    bbox = bbox_cache.ComputeWorldBound(mesh_prim)
    min_coords = bbox.GetBox().GetMin()
    max_coords = bbox.GetBox().GetMax()

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
    #import pdb; pdb.set_trace()
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
    
def rescale_scene(stage, scene_prim_root="/World/Scene", max_len=1):

    mesh_prim_path = get_all_mesh_prim_path(stage, scene_prim_root)
    print(mesh_prim_path)
    
    scale_factor = get_scale(stage, mesh_prim_path, max_len)
    #import pdb; pdb.set_trace()
    #centroid1 = get_centroid(stage, mesh_prim_path)
    #print("centriod: ", centroid1)
    # shift_x = -centroid[0]
    # shift_y = -centroid[1]
    # shift_z = -centroid[2]
    #shift_x = -centroid1[1]
    #shift_y = -centroid1[2]
    #shift_z = -centroid1[0]
    print(scale_factor)
    print(f"Scaling factor: {scale_factor}")


    # Apply the scaling to the mesh
    for prim_path in mesh_prim_path:

        mesh_prim = stage.GetPrimAtPath(prim_path)
            
        mesh = UsdGeom.Mesh(mesh_prim)
        '''
        points = mesh.GetPointsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()

        o3d_mesh = convert_to_open3d_mesh((points, face_vertex_counts, face_vertex_indices))
        o3d_mesh = rescale_mesh(o3d_mesh, scale_factor)
        
        # Extract mesh data
        vertices, faces = extract_mesh_data(o3d_mesh)
        
        # Create USD from the mesh data
        usd_file_path = "output_mesh.usd"
        create_usd_from_mesh(vertices, faces, usd_file_path)
        
        print(f"USD file saved to {usd_file_path}")
        import pdb; pdb.set_trace()
        
        '''
        if not mesh_prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} is not valid.")
        
        # Traverse up to the root ancestor
        #while mesh_prim.GetParent().GetPath()!= '/' and mesh_prim.GetPath() != '/':
        #    mesh_prim = mesh_prim.GetParent()
        mesh_prim = mesh_prim.GetParent().GetParent()
        xform = UsdGeom.Xformable(mesh_prim)

        #import pdb; pdb.set_trace()
        #shift_transform = Gf.Matrix4d().SetTranslate(Gf.Vec3d(shift_x, shift_y, shift_z))
        scale_transform = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
        
        xform.ClearXformOpOrder()  # Clear any existing transformations
        # Add translation first, then scale
        
        #shift_op = xform.AddTransformOp()
        #shift_op.Set(shift_transform)
        
        #xform.ClearXformOpOrder()
        scale_opsdfs = xform.AddTransformOp()
        #combined_transform = shift_transform
        combined_transform = scale_transform
        #combined_transform = scale_transform
        scale_opsdfs.Set(combined_transform)
        #import pdb; pdb.set_trace()
        
        centroid2 = get_centroid(stage, prim_path)
        #shift_x = -centroid2[1]
        #shift_y = -centroid2[2]
        #shift_z = -centroid2[0]
        
        shift_x = -centroid2[0]
        shift_y = -centroid2[1]
        shift_z = -centroid2[2]
        
        #shift_x = -centroid2[0]
        #shift_y = -centroid2[2]
        #shift_z = centroid2[1]
        print("centriod: ", centroid2)
        
        #print(centroid2)
        
        shift_transform = Gf.Matrix4d().SetTranslate(Gf.Vec3d(shift_x, shift_y, shift_z))
        xform.ClearXformOpOrder()
        shift_op = xform.AddTransformOp()
        combined_transform = shift_transform*scale_transform
        #combined_transform = scale_transform
        shift_op.Set(combined_transform)

        

    


