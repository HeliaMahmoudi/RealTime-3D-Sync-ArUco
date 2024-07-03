# Title:
Real-time 3D Model Animation in Blender Synchronization via ArUco Marker-Based Cube Position Estimation


## Overview:
The RealTime-3D-Sync project is designed to provide accurate and real-time synchronization between physical objects and 3D model animations. Leveraging OpenCV and Blender, this system uses ArUco marker detection to estimate the positions of real-world objects and seamlessly integrate these positions with corresponding 3D models in a virtual environment. By addressing the complex process of animation render clipping, this project enhances the realism and interaction of augmented reality (AR) applications, virtual production, gaming, and more.

## Demo:
````
<script src="https://cdn.jsdelivr.net/npm/can-autoplay@1.1.1/build/can-autoplay.js"></script>

<script>
  canAutoplay.video('https://www.youtube.com/watch?v=uTEGHKU78Dc', {
    inline: true,
    muted: true
  }).then(result => {
    if (result.result) {
      const iframe = document.createElement('iframe');
      iframe.src = 'https://www.youtube.com/embed/uTEGHKU78Dc?autoplay=1';
      iframe.width = '560';
      iframe.height = '315';
      iframe.frameBorder = '0';
      iframe.allowFullScreen = true;
      document.body.appendChild(iframe);
    } else {
      console.error(result.error);
    }
  });
</script>
````
- https://www.youtube.com/watch?v=uTEGHKU78Dc

## Features:

- ArUco marker Cube detection and tracking
- Real-time pose estimation using solvePnP
- Bundle adjustment for refining pose estimates
- Optical flow tracking for improved frame-to-frame consistency
- Exponential Moving Average (EMA) filtering for smoothing pose data
- UDP socket communication between OpenCV and Blender scripts
- Separate thread for continuous data reception in Blender 
- Real-time 3D model synchronization in Blender


## Requirements:
- Python 3.8+
- OpenCV 4.5+
- NumPy
- SciPy
- Blender 2.93+


## Usage: 
### Prepare the Calibration Data
Ensure you have the camera calibration data saved as calibration_data.npz. This file should include the camera_matrix and dist_coeffs. Place this file in the src/ directory.
### Setting up Blender
- Open Blender and create a new project.
- Add a 3D model (e.g., a cube.obj) to the scene and name it Cube.
- Open the src/blender_script.py script in Blender's text editor.
- Run the Blender script (src/blender_script.py) within Blender.
### Running the Capture and Processing Script
- Navigate to the src/ directory and run the OpenCV capture and processing script


## Note:
#### ArUco Dictionary and Marker IDs:
In this project, we used the DICT_6X6_250 ArUco dictionary and specific marker IDs [40, 23, 98, 124, 62, 203] in the desired order.

#### Recommended Dictionary for Best Performance:
- For most applications requiring fast and reliable detection, we recommend using the DICT_4X4_50 ArUco dictionary. This dictionary is smaller and simpler, making marker detection quicker and more robust, especially in real-time scenarios.
- Suggested marker IDs: Choose marker IDs that are not close to each other in numerical value to avoid potential confusion during detection. For example, [0, 5, 10, 15, 20, 25].


## Contact:
For any queries, reach out to [mahmoudi.helia@yahoo.com]


## Future Improvements:
- Multi-object tracking
- Improved occlusion handling
- Support for dynamic 3D models





