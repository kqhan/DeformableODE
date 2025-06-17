# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to work with randomized deformable objects and save vertex positions.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p ./data.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import random
import torch
import tqdm
import h5py
import os
import datetime
from PIL import Image

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on interacting with randomized deformable objects.")
parser.add_argument(
    "--save_camera",
    action="store_true",
    default=True,
    help="Save camera data to files.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils import convert_dict_to_backend


def define_camera_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    prim_utils.create_prim("/World/CameraOrigin", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/CameraOrigin/CameraSensor",
        update_period=0.0,
        height=1080,
        width=1920,
        data_types=[
            "rgb"
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)
    return camera


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = torch.rand(num_origins) * 0.5 + 0.5  # Random height between 1.0 and 1.5
    # return the origins
    return env_origins.tolist()


def design_scene():
    """Designs the scene with a bucket container and deformable objects falling into it."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Lights
    cfg_light = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light.func("/World/Light", cfg_light)

    # Create bucket container using rigid walls
    bucket_size = 0.8  # Reduced size of the bucket
    wall_thickness = 0.08
    wall_height = 0.6
    
    # Create bucket floor
    cfg_floor = sim_utils.CuboidCfg(
        size=(bucket_size, bucket_size, 0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # Make it static
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7)),
    )
    cfg_floor.func("/World/Bucket/Floor", cfg_floor, translation=[0.0, 0.0, 0.0])
    
    # Create front and back walls (oriented along Y-axis)
    cfg_wall_front_back = sim_utils.CuboidCfg(
        size=(wall_thickness, bucket_size, wall_height),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # Make walls static
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.6, 0.6, 0.8),
            opacity=1.0  # More opaque so walls are visible (70% opaque)
        ),
    )
    
    # Create left and right walls (oriented along X-axis)
    cfg_wall_left_right = sim_utils.CuboidCfg(
        size=(bucket_size, wall_thickness, wall_height),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # Make walls static
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.6, 0.6, 0.8),
            opacity=1.0  # More opaque so walls are visible (70% opaque)
        ),
    )
    
    # Position walls correctly
    # Front wall (positive X)
    cfg_wall_front_back.func("/World/Bucket/Wall_Front", cfg_wall_front_back, 
                           translation=[bucket_size/2 + wall_thickness/2, 0.0, wall_height/2])
    
    # Back wall (negative X)  
    cfg_wall_front_back.func("/World/Bucket/Wall_Back", cfg_wall_front_back,
                           translation=[-bucket_size/2 - wall_thickness/2, 0.0, wall_height/2])
    
    # Right wall (positive Y)
    cfg_wall_left_right.func("/World/Bucket/Wall_Right", cfg_wall_left_right,
                           translation=[0.0, bucket_size/2 + wall_thickness/2, wall_height/2])
    
    # Left wall (negative Y)
    cfg_wall_left_right.func("/World/Bucket/Wall_Left", cfg_wall_left_right,
                           translation=[0.0, -bucket_size/2 - wall_thickness/2, wall_height/2])

    # Define different deformable object configurations with better collision properties
    cfg_sphere = sim_utils.MeshSphereCfg(
        radius=0.08,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.0, 
            contact_offset=0.002,  # Slightly larger for better collision detection
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid = sim_utils.MeshCuboidCfg(
        size=(0.12, 0.12, 0.12),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.0, 
            contact_offset=0.002,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cylinder = sim_utils.MeshCylinderCfg(
        radius=0.06,
        height=0.2,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.0, 
            contact_offset=0.002,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_capsule = sim_utils.MeshCapsuleCfg(
        radius=0.06,
        height=0.2,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.0, 
            contact_offset=0.002,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cone = sim_utils.MeshConeCfg(
        radius=0.06,
        height=0.2,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.0, 
            contact_offset=0.002,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )

    # Create exactly 5 objects to fall into the bucket
    objects_cfg = [cfg_sphere, cfg_cuboid, cfg_cylinder, cfg_capsule, cfg_cone]
    
    # Spawn positions above the bucket (staggered heights to create interesting interactions)
    # Bucket is 0.8m x 0.8m, so safe zone is roughly ±0.3m from center
    # Making positions extremely centered to guarantee all objects fall in
    drop_positions = [
        [0.0, 0.0, 1.3],      # Center, lowest
        [0.05, 0.0, 1.1],     # Very slightly right, medium height
        [-0.05, 0.0, 1.2],    # Very slightly left, higher
        [0.0, 0.05, 1.3],     # Very slightly front, highest
        [0.0, -0.05, 1.4],    # Very slightly back, medium-high
    ]
    
    print("[INFO]: Spawning 5 deformable objects above the bucket...")
    
    # Create origin prims for each object
    for idx in range(5):
        prim_utils.create_prim(f"/World/Origin{idx}", "Xform", translation=drop_positions[idx])
    
    # Spawn each object with different properties
    for idx in range(5):
        obj_cfg = objects_cfg[idx]
        
        # Randomize material properties for bouncy behavior
        obj_cfg.physics_material.youngs_modulus = 891510.961825402  # Softer for more bounce
        # obj_cfg.physics_material.poissons_ratio = random.uniform(0.25, 0.45)
        
        # Assign distinct colors for easy identification
        colors = [
            (1.0, 0.3, 0.3),  # Red sphere
            (0.3, 1.0, 0.3),  # Green cube  
            (0.3, 0.3, 1.0),  # Blue cylinder
            (1.0, 1.0, 0.3),  # Yellow capsule
            (1.0, 0.3, 1.0),  # Magenta cone
        ]
        obj_cfg.visual_material.diffuse_color = colors[idx]
        
        # Spawn the object
        obj_cfg.func(f"/World/Origin{idx}/Object", obj_cfg, translation=[0.0, 0.0, 0.0])

    # Create a view for all the deformables
    cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Object",
        spawn=None,  # We already spawned the objects manually
        init_state=DeformableObjectCfg.InitialStateCfg(),
        debug_vis=True,
    )
    deformable_object = DeformableObject(cfg=cfg)

    # Create camera sensor
    camera = define_camera_sensor()

    # return the scene information
    scene_entities = {"deformable_object": deformable_object, "camera": camera}
    
    # Create object info for H5 file metadata
    object_info = [
        {"type": "sphere", "color": [1.0, 0.3, 0.3]},
        {"type": "cuboid", "color": [0.3, 1.0, 0.3]},
        {"type": "cylinder", "color": [0.3, 0.3, 1.0]},
        {"type": "capsule", "color": [1.0, 1.0, 0.3]},
        {"type": "cone", "color": [1.0, 0.3, 1.0]},
    ]
    
    return scene_entities, drop_positions, object_info


def create_vertex_data_file(filename: str, num_objects: int, object_info: list) -> h5py.File:
    """Create and initialize HDF5 file for vertex data storage."""
    h5_file = h5py.File(filename, 'w')
    
    # Create metadata group
    metadata = h5_file.create_group('metadata')
    metadata.attrs['num_objects'] = num_objects
    metadata.attrs['creation_time'] = datetime.datetime.now().isoformat()
    metadata.attrs['description'] = 'Deformable object vertex positions over time'
    
    # Store object information
    object_types = [info['type'] for info in object_info]
    colors = [info['color'] for info in object_info]
    
    metadata.create_dataset('object_types', data=[s.encode('utf-8') for s in object_types])
    metadata.create_dataset('object_colors', data=colors)
    
    # Create groups for time-series data
    h5_file.create_group('simulation_data')
    
    return h5_file


def save_vertex_data_to_h5(h5_file: h5py.File, deformable_object, sim_time: float, count: int):
    """Save vertex data to HDF5 file."""
    if h5_file is None:
        return
        
    # Create group for this timestep
    timestep_group = h5_file['simulation_data'].create_group(f'step_{count:06d}')
    timestep_group.attrs['sim_time'] = sim_time
    timestep_group.attrs['step_count'] = count
    
    # Get vertex positions and root positions
    nodal_pos = deformable_object.data.nodal_pos_w.cpu().numpy()
    root_pos = deformable_object.data.root_pos_w.cpu().numpy()
    
    # Save data for each object
    for obj_idx in range(deformable_object.num_instances):
        obj_group = timestep_group.create_group(f'object_{obj_idx}')
        
        # Save vertex positions
        vertices = nodal_pos[obj_idx]
        obj_group.create_dataset('vertex_positions', data=vertices)
        obj_group.create_dataset('root_position', data=root_pos[obj_idx])
        obj_group.attrs['num_vertices'] = vertices.shape[0]


def print_vertex_positions(deformable_object, sim_time: float, count: int, h5_file=None):
    """Print the positions of all vertices for each object and optionally save to H5."""
    if count % 20 == 0:  # Print every 20 steps to avoid too much output
        nodal_pos = deformable_object.data.nodal_pos_w
        print(f"\n=== Sim Time: {sim_time:.2f}s, Step: {count} ===")
        
        # Save to H5 file if provided
        if h5_file is not None:
            save_vertex_data_to_h5(h5_file, deformable_object, sim_time, count)
        
        for obj_idx in range(deformable_object.num_instances):
            print(f"Object {obj_idx}:")
            vertices = nodal_pos[obj_idx]
            print(f"  Num vertices: {vertices.shape[0]}")
            print(f"  Root position: {deformable_object.data.root_pos_w[obj_idx, :3]}")
            # Print first few vertices to avoid overwhelming output
            max_vertices_to_print = min(5, vertices.shape[0])
            for vert_idx in range(max_vertices_to_print):
                pos = vertices[vert_idx, :3]
                print(f"    Vertex {vert_idx}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            if vertices.shape[0] > max_vertices_to_print:
                print(f"    ... and {vertices.shape[0] - max_vertices_to_print} more vertices")
        
        if h5_file is not None:
            print(f"[INFO]: Vertex data saved to H5 file at step {count}")


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor, object_info: list):
    """Runs the simulation loop."""
    # Extract scene entities
    deformable_object = entities["deformable_object"]
    camera: Camera = entities["camera"]
    
    # Set camera position and target as specified
    camera_position = torch.tensor([[1.5, 1.5, 3.0]], device=sim.device)
    camera_target = torch.tensor([[0.0, 0.0, 0.4]], device=sim.device)
    camera.set_world_poses_from_view(camera_position, camera_target)
    
    # Verify camera position was set correctly
    print(f"[INFO]: Camera position set to: {camera_position[0]}")
    print(f"[INFO]: Camera target set to: {camera_target[0]}")
    
    # Create output directory for H5 files
    output_dir = "deformable_simulation_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output directory for camera images if saving is enabled
    camera_output_base_dir = None
    if args_cli.save_camera:
        camera_output_base_dir = os.path.join(output_dir, "camera_images")
        os.makedirs(camera_output_base_dir, exist_ok=True)
        print(f"[INFO]: Camera images will be saved to: {camera_output_base_dir}")
    
    # Create H5 file for this simulation run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_filename = os.path.join(output_dir, f"deformable_vertices_{timestamp}.h5")
    h5_file = create_vertex_data_file(h5_filename, deformable_object.num_instances, object_info)
    
    print(f"[INFO]: Saving vertex data to: {h5_filename}")
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    simulation_run = 0

    # Initialize camera output directory for first run
    camera_output_dir = None
    if args_cli.save_camera and camera_output_base_dir is not None:
        camera_output_dir = os.path.join(camera_output_base_dir, f"run_{simulation_run:02d}")
        os.makedirs(camera_output_dir, exist_ok=True)
        print(f"[INFO]: Camera images for run {simulation_run} will be saved to: {camera_output_dir}")

    # Nodal kinematic targets of the deformable bodies
    nodal_kinematic_target = deformable_object.data.nodal_kinematic_target.clone()

    try:
        # Simulate physics
        max_runs = 10
        print(f"[INFO]: Starting simulation for {max_runs} runs...")
        
        while simulation_app.is_running() and simulation_run < max_runs:
            # reset less frequently to let objects settle and interact in the bucket
            if count % 200 == 0:  # Increased from 400 to let objects interact longer
                # Close previous H5 file and create new one for next simulation run
                if count > 0:
                    h5_file.close()
                    simulation_run += 1
                    h5_filename = os.path.join(output_dir, f"deformable_vertices_{timestamp}_run{simulation_run:02d}.h5")
                    h5_file = create_vertex_data_file(h5_filename, deformable_object.num_instances, object_info)
                    print(f"[INFO]: Starting simulation run {simulation_run + 1}/{max_runs}, saving to: {h5_filename}")
                
                # Create separate folder for this simulation run's images
                if args_cli.save_camera and camera_output_base_dir is not None:
                    camera_output_dir = os.path.join(camera_output_base_dir, f"run_{simulation_run:02d}")
                    os.makedirs(camera_output_dir, exist_ok=True)
                    print(f"[INFO]: Camera images for run {simulation_run} will be saved to: {camera_output_dir}")
                
                # reset counters
                sim_time = 0.0
                count = 0

                # reset the nodal state of the objects back to their drop positions
                nodal_state = deformable_object.data.default_nodal_state_w.clone()
                
                # Reset objects to their original drop positions with slight randomization
                for obj_idx in range(deformable_object.num_instances):
                    # Very small random offset around original drop position to prevent drift outside bucket
                    pos_offset = torch.rand(3, device=sim.device) * 0.01  # Random offset ±0.01 (much smaller)
                    pos_w = torch.tensor(origins[obj_idx], device=sim.device) + pos_offset
                    pos_w = pos_w.unsqueeze(0)  # Add batch dimension
                    quat_w = math_utils.random_orientation(1, device=sim.device)
                    
                    # Transform nodal positions for this object
                    obj_nodal_state = nodal_state[obj_idx:obj_idx+1]
                    transformed_nodal_pos = deformable_object.transform_nodal_pos(
                        obj_nodal_state[..., :3], pos_w, quat_w
                    )
                    nodal_state[obj_idx, :, :3] = transformed_nodal_pos[0]

                # write nodal state to simulation
                deformable_object.write_nodal_state_to_sim(nodal_state)

                # Write the nodal state to the kinematic target and free all vertices
                nodal_kinematic_target[..., :3] = nodal_state[..., :3]
                nodal_kinematic_target[..., 3] = 1.0  # Free all vertices for natural physics
                deformable_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

                # reset buffers
                deformable_object.reset()

                # Reset camera position after simulation reset
                camera_position = torch.tensor([[1.5, 1.5, 3.0]], device=sim.device)
                camera_target = torch.tensor([[0.0, 0.0, 0.4]], device=sim.device)
                camera.set_world_poses_from_view(camera_position, camera_target)
                print(f"[INFO]: Camera position reset to {camera_position[0]} -> {camera_target[0]}")

                print("----------------------------------------")
                print(f"[INFO]: Run {simulation_run + 1}/{max_runs} - Dropping 5 deformable objects into the bucket...")

            # Minimal kinematic manipulation - let physics handle most interactions
            # Only apply very light manipulation to demonstrate kinematic control occasionally
            if count % 100 == 0 and count > 50:  # Apply gentle nudges occasionally
                pass
            else:
                # Keep all vertices free for natural bouncing
                nodal_kinematic_target[..., 3] = 1.0
                deformable_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            # write internal data to simulation
            deformable_object.write_data_to_sim()
            # perform step
            sim.step()
            # update sim-time
            sim_time += sim_dt
            count += 1
            # update buffers
            deformable_object.update(sim_dt)
            
            # Update camera
            camera.update(dt=sim_dt)
            
            # Save camera images if enabled (save every timestep)
            if args_cli.save_camera and camera_output_dir is not None:
                try:
                    # Get RGB images data
                    rgb_data = camera.data.output["rgb"][0].cpu().numpy()  # Shape: (H, W, 3)
                    
                    # Handle different possible RGB data ranges
                    if rgb_data.max() <= 1.0:
                        # Data is in [0, 1] range, convert to [0, 255]
                        rgb_image = (rgb_data * 255).astype(np.uint8)
                    else:
                        # Data might already be in [0, 255] range
                        rgb_image = rgb_data.astype(np.uint8)
                    
                    # Save images directly as RGB
                    image_filename = os.path.join(camera_output_dir, f"frame_{count:06d}.png")
                    pil_image = Image.fromarray(rgb_image, 'RGB')
                    pil_image.save(image_filename)
                    
                    # Print confirmation every 25 frames to avoid spam
                    if count % 25 == 0:
                        print(f"[INFO]: Saved camera image: {image_filename}")
                        print(f"       Camera position: {camera.data.pos_w[0]}")
                        
                except Exception as e:
                    print(f"[ERROR]: Failed to save camera image at step {count}: {e}")
            
            # Check and fix camera position if needed
            if count % 50 == 0:
                current_pos = camera.data.pos_w[0]
                if torch.allclose(current_pos, torch.zeros(3, device=sim.device), atol=0.1):
                    print(f"[WARNING]: Camera position is {current_pos}, resetting to [1.5, 1.5, 3.0]...")
                    camera_position = torch.tensor([[1.5, 1.5, 3.0]], device=sim.device)
                    camera_target = torch.tensor([[0.0, 0.0, 0.4]], device=sim.device)
                    camera.set_world_poses_from_view(camera_position, camera_target)
            
            # Print vertex positions for all objects and save to H5
            save_vertex_data_to_h5(h5_file, deformable_object, sim_time, count)
            
        print(f"[INFO]: Completed {simulation_run + 1} simulation runs successfully!")
        print(f"[INFO]: Data saved in {output_dir}/ directory")
    finally:
        # Ensure H5 file is properly closed
        if h5_file is not None:
            h5_file.close()
            print(f"[INFO]: H5 file closed successfully")


def main():
    """Main function."""
    # Load kit helper
    # sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / 120.0,  # high‑frequency physics
        # rendering_dt=1.0 / 60.0,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera for better view of the bucket scenario
    sim.set_camera_view([2.5, 2.5, 7.0], [0.0, 0.0, 0.4])  # type: ignore
    
    # Design scene
    scene_entities, scene_origins, object_info = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print(f"[INFO]: Created {scene_entities['deformable_object'].num_instances} deformable objects")
    print("[INFO]: Objects will fall into the bucket and bounce off each other!")
    print("[INFO]: Bucket dimensions: 0.8m x 0.8m x 0.6m height")
    print(f"[INFO]: Camera positioned at [2.5, 2.5, 3.0] looking at [0.0, 0.0, 0.4]")
    
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, object_info)  # type: ignore


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
