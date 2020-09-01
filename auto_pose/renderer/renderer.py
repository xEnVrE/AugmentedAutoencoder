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
        mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)

        # Create the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0])

        # Create and insert the camera
        self.camera = pyrender.IntrinsicsCamera(fx = fx, fy = fy, cx = cx, cy = cy, znear = near_plane, zfar = far_plane)
        self.scene.add(self.camera)

        # Create and insert the light
        self.light = pyrender.SpotLight(color = numpy.ones(3), intensity = 0.0)
        self.scene.add(self.light)

        # Create and insert a node for the object
        self.mesh_node = pyrender.Node(mesh = mesh, matrix = numpy.eye(4))
        self.scene.add_node(self.mesh_node)

        # Create a renderer
        self.renderer = pyrender.OffscreenRenderer(camera_width, camera_height)

    def render(self, R, z, random_light = False):

        # Update object pose
        object_pose = numpy.eye(4)
        object_pose[2, 3] = z
        object_pose[0:3, 0:3] = R
        self.scene.set_pose(self.mesh_node, object_pose)

        # Update light intensity
        intensity = 8.0
        if random_light:
            intensity = numpy.random.uniform(0.1, 1.0) * 60.0
        self.light.intensity = intensity

        # Render the scene
        render_rgb, render_depth = self.renderer.render(self.scene)

        # The AAE pipeline asks for BGR coded images
        render_rgb = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)

        return render_rgb, render_depth
