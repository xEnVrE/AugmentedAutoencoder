import numpy
import os
import sys
import trimesh

# Force rendering with EGL
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender

import cv2
class Renderer():
    def __init__(self, object_path, camera_width, camera_height, fx, fy, cx, cy, near_plane, far_plane):
        # Load the mesh
        trimesh_mesh = trimesh.load(object_path)
        self.mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)

        # Create and insert the camera
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera = pyrender.IntrinsicsCamera(fx = fx, fy = fy, cx = cx, cy = cy, znear = near_plane, zfar = far_plane)

    def render(self, R, z, random_light = False):
        # Create the scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0])

        # Insert the object
        object_pose = numpy.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ])
        object_pose[0:3, 0:3] = R
        scene.add(self.mesh, pose = object_pose)

        # Insert camera
        scene.add(self.camera)

        # Create and insert the light
        intensity = 8.0
        if random_light:
            intensity = numpy.random.uniform(0.1, 1.0) * 60.0
        light = pyrender.SpotLight(color = numpy.ones(3), intensity = intensity)
        scene.add(light)

        # Render the scene
        renderer = pyrender.OffscreenRenderer(self.camera_width, self.camera_height)
        render_rgb, render_depth = renderer.render(scene)

        # The AAE pipeline asks for BGR coded images
        render_rgb = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)

        return render_rgb, render_depth
