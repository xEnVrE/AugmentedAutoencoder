import glob
import numpy
import os
import random
import sys
import trimesh
from scipy.spatial.transform import Rotation as scipyR

# Force rendering with EGL
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender

import cv2
class Renderer():
    def __init__(self, objects_path, object_file, camera_width, camera_height, fx, fy, cx, cy, near_plane, far_plane):

        object_path = objects_path + "/" + object_file
        meshes_path = [path for path in glob.glob(objects_path + "/*.ply") if object_file not in path]

        # Load the meshes
        mesh = pyrender.Mesh.from_trimesh(trimesh.load(object_path))
        other_meshes = [pyrender.Mesh.from_trimesh(trimesh.load(path)) for path in meshes_path]

        # Create the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0])

        # Create and insert the camera
        self.camera = pyrender.IntrinsicsCamera(fx = fx, fy = fy, cx = cx, cy = cy, znear = near_plane, zfar = far_plane)
        self.scene.add(self.camera)

        # Create and insert the light
        self.light = pyrender.DirectionalLight(color = numpy.ones(3), intensity = 0.0)
        self.scene.add(self.light)
        self.light_color_var = 0.8

        # Create and insert a node for the object
        self.mesh_node = pyrender.Node(mesh = mesh, matrix = numpy.eye(4))
        self.scene.add_node(self.mesh_node)

        # Create and insert nodes for the other objects
        self.hidden_pose = numpy.eye(4)
        self.hidden_pose[2, 3] = 1.0
        self.other_mesh_nodes = []
        for mesh in other_meshes:
            mesh_node = pyrender.Node(mesh = mesh, matrix = self.hidden_pose)
            self.scene.add_node(mesh_node)
            self.other_mesh_nodes.append(mesh_node)

        # Create a renderer
        self.renderer = pyrender.OffscreenRenderer(camera_width, camera_height)

        self.random_z_probability = 0.5
        self.number_other_objects = 2
        self.other_objects_probability = 0.5

    def render(self, R, z):
        # Always reset poses of other objects
        for node in self.other_mesh_nodes:
            self.scene.set_pose(node, self.hidden_pose)

        # Set default object pose
        object_pose = numpy.eye(4)
        object_pose[2, 3] = z
        object_pose[0:3, 0:3] = R
        self.scene.set_pose(self.mesh_node, object_pose)

        # Set default light intensity and color
        self.light.color = numpy.ones(3)
        self.light.intensity = 40.0

        # Render the scene
        render_rgb, render_depth = self.renderer.render(self.scene)
        render_rgb = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)

        # Sample random z
        if numpy.random.random() < self.random_z_probability:
            scaler = numpy.random.uniform(0.9, 1.1)
            z = z / scaler
        object_pose[2, 3] = z
        self.scene.set_pose(self.mesh_node, object_pose)

        # Sample random light intensity
        self.light.intensity = numpy.random.uniform(0.1, 1.0) * 100.0

        # Sample random light color
        self.light.color = numpy.array([numpy.random.uniform(1 - self.light_color_var, 1 + self.light_color_var),
                                        numpy.random.uniform(1 - self.light_color_var, 1 + self.light_color_var),
                                        numpy.random.uniform(1 - self.light_color_var, 1 + self.light_color_var)])

        # Render the other objects
        if numpy.random.random() < self.other_objects_probability:
            # Add objects at random to simulate occlusions
            is_in_center = True
            for i in random.choices(range(len(self.other_mesh_nodes)), k = self.number_other_objects):
                # Sample random rotation
                rotation = scipyR.random().as_matrix()
                pose = numpy.eye(4)
                pose[0:3, 0:3] = rotation

                # Sample random position
                offset_x = 0.08 + random.uniform(-0.05, 0.0)
                if numpy.random.random() < 0.5:
                    offset_x = -0.08 + random.uniform(0.0, 0.05)

                offset_y = 0.08 + random.uniform(-0.05, 0.0)
                if numpy.random.random() < 0.5:
                    offset_y = -0.08 + random.uniform(0.0, 0.05)

                x = offset_x
                y = offset_y

                # if is_in_center:
                #     x = 0
                #     y = 0
                #     is_in_center = False

                pose[0, 3] = x
                pose[1, 3] = y
                pose[2, 3] = z + random.uniform(0.0, 0.03)

                self.scene.set_pose(self.other_mesh_nodes[i], pose)

        # Render the scene
        render_rgb_aug, render_depth_aug = self.renderer.render(self.scene)
        render_rgb_aug = cv2.cvtColor(render_rgb_aug, cv2.COLOR_RGB2BGR)

        return render_rgb, render_depth, render_rgb_aug, render_depth_aug
