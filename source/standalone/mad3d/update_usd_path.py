import argparse
from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Utility to create occupancy grid")
# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import glob
import numpy as np
import torch
import omni
import omni.isaac.lab.sim as sim_utils
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.lab.sensors import CameraCfg, Camera
from omni.isaac.lab.utils.math import transform_points, unproject_depth
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from pxr import Usd, Sdf

def update_texture_path(usd_file, old_prefix, new_prefix):
    stage = Usd.Stage.Open(usd_file)
    print(usd_file)
    # Iterate through all prims in the stage
    updated = False
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            # Check if the attribute is a string or path
            value = attr.Get()
            if isinstance(value, str) and old_prefix in value:
                # Replace the old prefix with the new prefix
                new_value = value.replace(old_prefix, new_prefix, 1)
                attr.Set(new_value)
                print(f"Updated: {value} -> {new_value}")
                updated = True

    # Save the stage if any updates were made
    if updated:
        stage.GetRootLayer().Save()
        print(f"Saved updates to: {usd_file}")
    else:
        print(f"No updates needed in: {usd_file}")



# Paths to update
old_prefix = "/home/dsr/Documents/mad3d/New_Dataset20/house3k/"
new_prefix = "/home/Dataset/houes3k_env20/"

# Update USD files in a directory
search_dir = "/home/Dataset/houes3k_env20/preprocess/BATCH_2/Set_A/BAT2_SETA_HOUSE1"
for root, _, files in os.walk(search_dir):
    print(root)
    for file in files:
        if file.endswith(".usd") or file.endswith(".usda"):
            usd_file = os.path.join(root, file)
            update_texture_path(usd_file, old_prefix, new_prefix)

