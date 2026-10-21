import os
import sys
import json
import numpy as np
import mujoco
from PIL import Image

sys.path.append(os.path.dirname(__file__))
from models import ProposedVM
from quadruped_env import QuadrupedEnv

def capture():
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "data", "experimental_results.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(base_dir, "experimental_results.json")
        
    target_dir = os.path.join(base_dir, "figures")
    if not os.path.exists(target_dir):
        target_dir = base_dir
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    model = ProposedVM()
    model.set_params(np.array(data["experiment_1_learning"]["proposed"]["best_params"]))
    
    env = QuadrupedEnv(max_steps=250)
    obs = env.reset(randomize=False)
    
    width, height = 1280, 720
    renderer = mujoco.Renderer(env.model, height, width)
    
    # Cameras
    cam_iso = mujoco.MjvCamera()
    cam_iso.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_iso.elevation = -18
    cam_iso.azimuth = 135
    cam_iso.distance = 1.3
    
    cam_side = mujoco.MjvCamera()
    cam_side.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_side.elevation = -10
    cam_side.azimuth = 90
    cam_side.distance = 1.25
    
    cam_front = mujoco.MjvCamera()
    cam_front.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_front.elevation = -12
    cam_front.azimuth = 180
    cam_front.distance = 1.3
    
    gif_frames = []
    
    print("Running simulation and capturing frames...")
    for step in range(160):
        tau = model.compute_torque(obs)
        obs, r, done, info = env.step(tau)
        
        # When robot enters steady trot gait
        if step == 95:
            # 1. Isometric View
            cam_iso.lookat = env.data.qpos[:3]
            renderer.update_scene(env.data, camera=cam_iso)
            img_iso = Image.fromarray(renderer.render())
            p_iso = os.path.join(target_dir, "mujoco_screenshot_isometric.png")
            img_iso.save(p_iso)
            print(f"Saved: {p_iso}")
            
            # 2. Side View
            cam_side.lookat = env.data.qpos[:3]
            renderer.update_scene(env.data, camera=cam_side)
            img_side = Image.fromarray(renderer.render())
            p_side = os.path.join(target_dir, "mujoco_screenshot_side.png")
            img_side.save(p_side)
            print(f"Saved: {p_side}")
            
            # 3. Front View
            cam_front.lookat = env.data.qpos[:3]
            renderer.update_scene(env.data, camera=cam_front)
            img_front = Image.fromarray(renderer.render())
            p_front = os.path.join(target_dir, "mujoco_screenshot_front.png")
            img_front.save(p_front)
            print(f"Saved: {p_front}")
            
        # Collect frames for animated GIF during steady walking (steps 80 to 140, approx 1 trot cycle)
        if 80 <= step <= 140 and step % 2 == 0:
            cam_iso.lookat = env.data.qpos[:3]
            renderer.update_scene(env.data, camera=cam_iso)
            # Resize for GIF to keep file size reasonable
            frame_img = Image.fromarray(renderer.render()).resize((640, 360), Image.Resampling.LANCZOS)
            gif_frames.append(frame_img)
            
    if gif_frames:
        p_gif = os.path.join(target_dir, "mujoco_walking.gif")
        gif_frames[0].save(
            p_gif,
            save_all=True,
            append_images=gif_frames[1:],
            duration=40,  # 25 fps
            loop=0
        )
        print(f"Saved animated GIF: {p_gif}")
        
    print("All captures completed!")

if __name__ == "__main__":
    capture()
