from math import sqrt
from typing import Optional, Tuple

from isaacgym import gymapi, gymutil
from isaacgym.terrain_utils import *
import numpy as np
import torch

def create_simulation(
    gym,
    dt: Optional[float] = 0.005,
    gravity: Optional[float] = -9.81,
    use_gpu_pipeline: Optional[bool] = False
):
    '''Create a simulation.
    
    Create a `sim` object with `gymapi.SimParams`.

    Args
    ----
    gym: a `gym` object.
    dt: (optional) float
        simulation step size.
    gravity: (optional) float
        z-component for the 3-d vector representing 
        gravity force in Newtons.
    use_gpu_pipeline: (optional) bool
        # TODO

    Returns
    -------
    a `sim` object with `gymapi.SimParams`.
    '''
    # get default set of parameters
    sim_params = gymapi.SimParams()

    # set common parameters
    sim_params.dt = dt
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, gravity)

    # set PhysX-specific parameters
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = use_gpu_pipeline
    sim_params.physx.solver_type = 1  # TGS
    sim_params.physx.num_position_iterations = 4  # 4 improve solver convergence
    sim_params.physx.num_velocity_iterations = 0
    # shapes whose distance is less than the sum of their 
    # contactOffset values will generate contacts
    sim_params.physx.contact_offset = 0.001
    # two shapes will come to rest at a distance equal to 
    # the sum of their restOffset values
    sim_params.physx.rest_offset = 0.0
    # A contact with a relative velocity below this will not bounce.
    sim_params.physx.bounce_threshold_velocity = 0.5
    # The maximum velocity permitted to be introduced by the solver 
    # to correct for penetrations in contacts.
    sim_params.physx.max_depenetration_velocity = 1.0

    # create sim with these parameters
    sim = gym.create_sim(
        compute_device=0,
        graphics_device=0, 
        type=gymapi.SIM_PHYSX, 
        params=sim_params
    )

    return sim


def creat_ground_plane(
    gym, 
    sim, 
    static_friction: Optional[float] = 1.0, 
    dynamic_friction: Optional[float] = 1.0
) -> None:
    '''Create a ground plane for simulation.

    Args
    ----
    gym: a `gym` object.
    sim: a `sim` object.
    static_friction: (optional) float
        coefficients of static friction.
    dynamic friction: (optional) float
        coefficients of dynamic friction.
    '''
    plane_params = gymapi.PlaneParams()

    # the orientation of the plane.
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    # defines the distance of the plane from the origin.
    plane_params.distance = 0
    plane_params.static_friction = static_friction
    plane_params.dynamic_friction = dynamic_friction
    # used to control the elasticity of collisions with the 
    # ground plane (amount of bounce).
    plane_params.restitution = 0

    # create the ground plane
    gym.add_ground(sim, plane_params)


def load_model(
    gym, 
    sim, 
    root_path: str, 
    file_path: str, 
    is_base_fixed: Optional[bool] = False
):
    '''Loading a robot model.

    Gym currently supports loading URDF, MJCF, and USD file formats. 
    Loading an asset file creates a GymAsset object that includes 
    the definiton of all the bodies, collision shapes, visual attachments, 
    joints, and degrees of freedom (DOFs). 

    Args
    ----
    gym: a `gym` object.
    sim: a `sim` object.
    root_path: str
        asset root directory, can be specified as an absolute 
        path or as a path relative to the current working directory.
    file_path: str
        the path of the model (Ex. the urdf file) relative 
        to the `root_path`.
    is_base_fixed: (optional) bool 
        is base link fixed.

    Returns
    -------
    a `GymAsset` object that includes the definiton of all the bodies, 
    collision shapes, visual attachments, joints, and degrees of freedom (DOFs). 
    '''
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options.fix_base_link = is_base_fixed
    asset_options.use_mesh_materials = True
    asset_options.flip_visual_attachments = True
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    
    # added to the diagonal elements of inertia tensors for all of the 
    # asset’s rigid bodies/links. could improve simulation stability.
    asset_options.armature = 0.01   

    asset_options.use_mesh_materials = True
    robot_model = gym.load_asset(sim, root_path, file_path, asset_options)
    return robot_model


def create_envs_actors(
    gym, 
    sim, 
    robot_model, 
    body_height: float, 
    num_envs: Optional[int] = 1, 
    envs_per_row: Optional[int] = -1, 
    env_spacing: Optional[float] = 1.0, 
    actor_name: Optional[str] = 'Actor'
) -> Tuple[list, list]:
    '''Create environments and actors.

    An environment consists of a collection of actors and sensors 
    that are simulated together. Actors within an environment 
    interact with each other physically. Their state is maintained 
    by the physics engine and can be controlled using the control API. 
    Sensors placed in an environment, like cameras, will be able 
    to capture the actors in that environment.

    Each env has its own coordinate space, which gets embedded in the 
    global simulation space. When creating an environment, we specify 
    the local extents of the environment, which depend on the desired 
    spacing between environment instances. As new environments get 
    added to the simulation, they will be arranged in a 2D grid one 
    row at a time.

    An actor is simply an instance of a `GymAsset`. Each actor must 
    be placed in an environment. You cannot have an actor that doesn't 
    belong to an environment. The actor pose is defined in env-local 
    coordinates using position vector p = (0, 0, body_height), and 
    orientation quaternion r = I.

    Args
    ----
    gym: a `gym` object.
    sim: a `sim` object.
    robot_model: a robot model object.
    body_height: float
        z-component of the base position expressed in world frame.
    num_envs: (optional) int
        number of environments.
    envs_per_row: (optional) int
        number of environments per row.
    env_spacing: (optional) float
        desired space between environment instances.
    actor_name: (optional) str
        specify a name for your actor. this makes it possible to 
        look up the actor by name. If you wish to do this, make sure 
        that you assign unique names to all your actors within the 
        same environment.

    Returns
    -------
    two lists. the one containing environment instances, the other 
    containing actor handles.
    '''
    
    if envs_per_row == -1:
        envs_per_row = int(sqrt(num_envs))

    # set up the env grid
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing*2, env_spacing, env_spacing)

    envs = []
    actor_handles = []

    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        envs.append(env)

        pose = gymapi.Transform()
        init_pos_base = torch.zeros(3, dtype=torch.float32, device='cuda:0')
        init_pos_base[2] = body_height
        pose.p = gymapi.Vec3(*init_pos_base)

        # pose.p = gymapi.Vec3(0.0, 0.0, body_height)

        cur_actor_name = actor_name + str(i)

        actor_handle = gym.create_actor(env, robot_model, pose, cur_actor_name, 
            group=i, filter=1)
        actor_handles.append(actor_handle)

    return envs, actor_handles


def add_viewer(gym, sim):
    '''Create a viewer.

    By default, the simulation does not create any visual feedback window. 
    This allows the simulations to run on headless workstations or clusters 
    with no monitors attached. When developing and testing, however, it is 
    useful to be able to visualize the simulation. Isaac Gym comes with a 
    simple integrated viewer that lets you see what’s going on in the 
    simulation.

    Args
    ----
    gym: a `gym` object.
    sim: a `sim` object.
    
    Returns
    -------
    a viewer that lets you see what's going on the simulation.
    '''
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)

    return viewer



def add_terrain(gym, sim, name="slope", x_offset=2., invert=False, width=2.8):
    # terrains
    num_terrains = 1
    terrain_width = 2.
    terrain_length = width
    horizontal_scale = 0.05  # [m] resolution in x
    vertical_scale = 0.005  # [m] resolution in z
    num_rows = int(terrain_width/horizontal_scale)
    num_cols = int(terrain_length/horizontal_scale)
    heightfield = np.zeros((num_terrains*num_rows, num_cols), dtype=np.int16)

    step_height = 0.07
    step_width = 0.3
    num_steps = terrain_width / step_width
    height = step_height * num_steps
    slope = height / terrain_width
    # num_stairs = height / step_height
    # step_width = terrain_length / num_stairs
    
    def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
    if name=="slope":
        heightfield[0: num_rows, :] = sloped_terrain(new_sub_terrain(), slope=slope).height_field_raw
    elif name=="stair":
        heightfield[0: num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=step_width, step_height=step_height).height_field_raw
    elif name=="pyramid":
        heightfield[0: num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=step_width, step_height=step_height).height_field_raw
    else:
        raise NotImplementedError("Not support terrains!")

    if invert:
        heightfield[0: num_rows, :] = heightfield[0: num_rows, :][::-1]

    # add the terrain as a triangle mesh
    vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    tm_params.transform.p.x = x_offset
    tm_params.transform.p.y = -1
    if name=="stair":
        tm_params.transform.p.z = -0.09
    elif name=="pyramid":
        tm_params.transform.p.z = 0.01

    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)
    return heightfield