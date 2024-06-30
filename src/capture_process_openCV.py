import cv2
import numpy as np
import cv2.aruco as aruco
import socket
import struct
from scipy.optimize import least_squares
# import time

# Load camera calibration data
calibration_data = np.load('calibration_data.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parameters.adaptiveThreshConstant = 7
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.03
parameters.minCornerDistanceRate = 0.05
parameters.minMarkerDistanceRate = 0.05
parameters.minOtsuStdDev = 5.0
parameters.errorCorrectionRate = 0.6

# Set up video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS as it is the supported rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Verify the frame rate
actual_rate = cap.get(cv2.CAP_PROP_FPS)
print(f"Set frame rate: 30, Actual frame rate: {actual_rate}")

# Define the marker points for each face of the cube
cube_edge_size = 0.06
marker_size = 0.052

c_pt = cube_edge_size / 2
m_pt = marker_size / 2

# Define the ArUco board parameters
marker_ids = np.array([40, 23, 98, 124, 62, 203], dtype=np.int32)
marker_points = [
    np.array([[-m_pt, m_pt, c_pt], [m_pt, m_pt, c_pt], [m_pt, -m_pt, c_pt], [-m_pt, -m_pt, c_pt]], dtype=np.float32),
    np.array([[-m_pt, -c_pt, m_pt], [m_pt, -c_pt, m_pt], [m_pt, -c_pt, -m_pt], [-m_pt, -c_pt, -m_pt]], dtype=np.float32),
    np.array([[-c_pt, m_pt, m_pt], [-c_pt, -m_pt, m_pt], [-c_pt, -m_pt, -m_pt], [-c_pt, m_pt, -m_pt]], dtype=np.float32),
    np.array([[m_pt, c_pt, m_pt], [-m_pt, c_pt, m_pt], [-m_pt, c_pt, -m_pt], [m_pt, c_pt, -m_pt]], dtype=np.float32),
    np.array([[c_pt, -m_pt, m_pt], [c_pt, m_pt, m_pt], [c_pt, m_pt, -m_pt], [c_pt, -m_pt, -m_pt]], dtype=np.float32),
    np.array([[-m_pt, -m_pt, -c_pt], [m_pt, -m_pt, -c_pt], [m_pt, m_pt, -c_pt], [-m_pt, m_pt, -c_pt]], dtype=np.float32)
]

def project_points(points, rvec, tvec):
    """Project 3D points to 2D using the camera parameters"""
    points_2d, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

def reprojection_error(params, points_3d, points_2d):
    """Calculate reprojection error"""
    rvec = params[:3]
    tvec = params[3:]
    projected = project_points(points_3d, rvec, tvec)
    return (projected - points_2d).ravel()

def bundle_adjustment(points_3d, points_2d, rvec, tvec):
    """Perform bundle adjustment to refine pose estimate"""
    params = np.hstack((rvec.ravel(), tvec.ravel()))
    result = least_squares(reprojection_error, params, args=(points_3d, points_2d))
    return result.x[:3].reshape(3, 1), result.x[3:].reshape(3, 1)

def calculate_optical_flow(prev_gray, gray, prev_points):
    """Calculate optical flow using Lucas-Kanade method"""
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
    good_new = new_points[status == 1]
    good_old = prev_points[status == 1]
    return good_new, good_old

def track_points(prev_frame, frame, prev_points):
    """Track points using optical flow"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return calculate_optical_flow(prev_gray, gray, prev_points)

def update_3d_2d_correspondences(obj_points, img_points, new_img_points):
    """Update 3D-2D correspondences based on optical flow results"""
    updated_obj_points = []
    updated_img_points = []
    for i, (old_pt, new_pt) in enumerate(zip(img_points, new_img_points)):
        if np.linalg.norm(old_pt - new_pt) < 10:  # Threshold for maximum allowed movement
            updated_obj_points.append(obj_points[i])
            updated_img_points.append(new_pt)
    return np.array(updated_obj_points), np.array(updated_img_points)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Set up the socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 12345)

prev_frame = None
prev_obj_points = None
prev_img_points = None
prev_rvec = None
prev_tvec = None

# EMA smoothing factors
alpha_rvec = 0.2  # Adjust this value between 0 and 1 for rvec smoothing
alpha_tvec = 1  # Adjust this value between 0 and 1 for tvec smoothing

smoothed_rvec = None
smoothed_tvec = None

# Define the full cube points (vertices of the cube)
cube_points = np.array([
    [-c_pt, -c_pt, -c_pt],
    [-c_pt, c_pt, -c_pt],
    [c_pt, c_pt, -c_pt],
    [c_pt, -c_pt, -c_pt],
    [-c_pt, -c_pt, c_pt],
    [-c_pt, c_pt, c_pt],
    [c_pt, c_pt, c_pt],
    [c_pt, -c_pt, c_pt]
], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        ids = ids.flatten()
        print(f"Detected markers: {ids}")

        obj_points = []
        img_points = []

        for i, id in enumerate(ids):
            if id in marker_ids:
                idx = np.where(marker_ids == id)[0][0]
                obj_points.extend(marker_points[idx])
                img_points.extend(corners[i][0])

        if len(obj_points) > 0:
            obj_points = np.array(obj_points, dtype=np.float32)
            img_points = np.array(img_points, dtype=np.float32)

            print(f"Number of point correspondences: {len(obj_points)}")

            try:
                if prev_rvec is not None and prev_tvec is not None:
                    _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs, prev_rvec, prev_tvec, useExtrinsicGuess=True)
                else:
                    _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
                print("solvePnP successful")

                rvec_refined, tvec_refined = bundle_adjustment(obj_points, img_points, rvec, tvec)
                print("Bundle adjustment successful")

                # Apply EMA filter
                if smoothed_rvec is None:
                    smoothed_rvec = rvec_refined
                else:
                    smoothed_rvec = alpha_rvec * rvec_refined + (1 - alpha_rvec) * smoothed_rvec

                if smoothed_tvec is None:
                    smoothed_tvec = tvec_refined
                else:
                    smoothed_tvec = alpha_tvec * tvec_refined + (1 - alpha_tvec) * smoothed_tvec

                aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, smoothed_rvec, smoothed_tvec, 0.07)
                print("Axes drawn")

                # Project and draw the full cube
                imgpts, _ = cv2.projectPoints(cube_points, smoothed_rvec, smoothed_tvec, camera_matrix, dist_coeffs)
                imgpts = np.int32(imgpts).reshape(-1, 2)

                # Draw the edges of the cube
                frame = cv2.drawContours(frame, [imgpts[:4]], -1, (255, 0, 0), 1)  # Draw the bottom face in red
                frame = cv2.drawContours(frame, [imgpts[4:]], -1, (255, 0, 0), 1)  # Draw the top face in green
                for i, j in zip(range(4), range(4, 8)):
                    frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 1)  # Draw vertical edges in blue
                print("Cube drawn")

                data = struct.pack('!6f', *smoothed_rvec.flatten(), *smoothed_tvec.flatten())
                print(f"Sending data: {data}")
                sock.sendto(data, server_address)

                # Update previous pose estimation
                prev_rvec = rvec_refined
                prev_tvec = tvec_refined
            except Exception as e:
                print(f"Error in pose estimation or drawing: {e}")
    else:
        print("No markers detected")

    # Flip the frame (optional, depends on your camera setup)
    frame = cv2.flip(frame, flipCode=1)  # 1 for horizontal flip, 0 for vertical flip

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
