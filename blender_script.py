import bpy
import socket
import struct
import threading
import mathutils
import numpy as np
import cv2

# Socket setup to receive data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('localhost', 12345))

# Filtering parameters
alpha_rvec = 0.2  # Smoothing factor for EMA filter for rotation vector
alpha_tvec = 0.2  # Smoothing factor for EMA filter for translation vector
prev_rvec = None
prev_tvec = None

def unpack_pose_data(data):
    """ Function to convert received data into usable format """
    return struct.unpack('!6f', data)

def apply_ema_filter(new_val, prev_val, alpha=0.2):
    """ Apply Exponential Moving Average filter """
    if prev_val is None:
        return new_val
    return alpha * new_val + (1 - alpha) * prev_val

def transform_coordinates(rvec, tvec):
    """ Transform coordinates from OpenCV to Blender """
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Swap Y and Z axes, and invert Z axis
    transformation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    # Apply transformation
    rotation_matrix = transformation_matrix @ rotation_matrix @ transformation_matrix.T
    tvec = transformation_matrix @ tvec

    return rotation_matrix, tvec

def update_cube_pose(scene):
    global prev_rvec, prev_tvec
    try:
        data, addr = sock.recvfrom(1024)
        pose_data = unpack_pose_data(data)
        rvec, tvec = np.array(pose_data[:3]), np.array(pose_data[3:])

        # Apply filtering
        rvec = apply_ema_filter(rvec, prev_rvec, alpha_rvec)
        tvec = apply_ema_filter(tvec, prev_tvec, alpha_tvec)

        # Update previous values
        prev_rvec = rvec
        prev_tvec = tvec

        # Transform coordinates from OpenCV to Blender
        rotation_matrix, tvec = transform_coordinates(rvec, tvec)

        # Apply the transformation to the 3D model in Blender
        cube = bpy.data.objects['Cube']
        cube.location = tvec.tolist()

        # Convert rotation matrix to Euler angles
        rotation_euler = mathutils.Matrix(rotation_matrix).to_euler()
        cube.rotation_mode = 'XYZ'
        cube.rotation_euler = rotation_euler

    except Exception as e:
        print(f"Error updating pose: {e}")

def receive_data():
    while True:
        update_cube_pose(None)

# Start receiving data in a separate thread
thread = threading.Thread(target=receive_data)
thread.daemon = True
thread.start()

# Ensure the Blender script is updated regularly
def update(scene):
    pass

# Add the update function to Blender's frame change handler
bpy.app.handlers.frame_change_pre.append(update)
