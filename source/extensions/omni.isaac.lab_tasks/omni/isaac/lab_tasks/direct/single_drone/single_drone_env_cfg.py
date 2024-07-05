import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.actuators import ImplicitActuatorCfg


CRAZYFLIE_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, # won't drop
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
        copy_from_source=False, # mirror
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

    # env
    env_size = 20
    grid_size = 20
    preplan = False
    save_img = True
    vis_occ = False
    save_env_ids = [0, 1]

    num_envs = 8 # this might be overwrote by parser
    env_spacing = 30 # in meter, 2 cells is one unit

    decimation = 5 # _apply_action will run decimation time
    num_actions = 5 # x, y, z, yaw, pitch
    num_observations = 350000 # rgb image, occ grid, drone pose
    num_states = 0
    episode_length_s = 500 # timeout
    debug_vis = False

    # obv
    img_t = 2 
    total_img = 50000

    # occ grid
    decrement=0.4
    increment=0.84
    max_log_odds=3.5
    min_log_odds=-3.5

    # sensor
    camera_offset = [0.0, 0.0, -0.2]
    camera_w, camera_h = 200, 200

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/50, # physical simulation step
        disable_contact_processing=True,
        physx=sim_utils.PhysxCfg(use_gpu=True), # with use_gpu, the buffer cannot grow dynamically
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        gravity=(0.0, 0.0, 0.0), # disable gravity
        enable_scene_query_support=False, # disable collision query
        render_interval=decimation
    )

    # ground	
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
    
    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG

    # sensor    
    camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(pos=camera_offset, convention="world"),
        update_period=0, # update every physical step
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            #focal_length=1.38, # in cm
            #focus_distance=1.0, # in m 
            #horizontal_aperture=24., # in mm 
            clipping_range=(0.1, 20.0) # near and far plane in meter
        ),
        width=camera_w,
        height=camera_h,
      
    )
   
    # setup interactive scene for rl training
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, 
                                                     env_spacing=env_spacing,
                                                     lazy_sensor_update=False, # update sensor data every time
                                                     replicate_physics=True)
