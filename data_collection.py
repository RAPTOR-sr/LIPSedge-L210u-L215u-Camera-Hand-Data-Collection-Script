from openni import openni2
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
import mediapipe as mp
import math
import os
import json
from sklearn.neighbors import NearestNeighbors



# Define the point data structure
class PointData:
    def __init__(self):
        self.depth = 0
        self.worldX = 0.0
        self.worldY = 0.0
        self.worldZ = 0.0
        self.r = 0.0
        self.g = 0.0
        self.b = 0.0

# Define the viewer state (for camera controls)
class ViewerState:
    def __init__(self):
        self.yaw = 0
        self.pitch = 0
        self.lastX = 0
        self.lastY = 0
        self.offset = -2.0
        self.lookatX = 0.0
        self.lookatY = 0.0
        self.mouseLeft = False
        self.mouseRight = False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==== Global variables ====

hand_regions = [] # Global variable to store hand landmarks
recording = False
recording_frames = 0
max_recording_frames = 15
recording_data = []
Class_id = ""
User_id = ""


try:
    openni2.initialize("C:\\Program Files\\LIPSedge Camera SDK 1.02\\LIPSedge L210 2.4.4.3_v1.6.5\\OpenNI2\\Redist")
    dev = openni2.Device.open_any()

    # Create color and depth streams
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    ir = dev.create_ir_stream()

    # Enable depth and color frame synchronization
    dev.set_depth_color_sync_enabled(True)

    ir.start()
    depth_stream.start()
    color_stream.start()
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit(1)

# Camera intrinsics (adjust if needed)
fx, fy = 415.774663, 416.338525
scale = 1000.0

def create_data_dirs(base_path, class_id, user_id):
    paths = {
        "rgb": os.path.join(base_path, "rgb", class_id, user_id),
        "depth": os.path.join(base_path, "depth", class_id, user_id),
        "amplitude": os.path.join(base_path, "apltitude", class_id, user_id),  # Note the typo here
        "point_cloud": os.path.join(base_path, "point_cloud", class_id, user_id),
        "full_point_cloud": os.path.join(base_path, "full_point_cloud", class_id, user_id)
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

# Callback functions for keyboard and mouse interactions
def mouse_button_callback(window, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT:
        viewerState.mouseLeft = action == glfw.PRESS
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        viewerState.mouseRight = action == glfw.PRESS

def scroll_callback(window, xoffset, yoffset):
    viewerState.offset += yoffset * 0.1

def cursor_position_callback(window, xpos, ypos):
    if viewerState.mouseLeft:
        viewerState.yaw += xpos - viewerState.lastX
        viewerState.pitch += ypos - viewerState.lastY
    viewerState.lastX = xpos
    viewerState.lastY = ypos

def key_callback(window, key, scancode, action, mods):
    global recording, recording_frames, Class_id, User_id

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    elif key == glfw.KEY_R and action == glfw.PRESS:
        Class_id = input("Enter Class: ")
        User_id = input("Enter User: ")
        print(f"Recording started for Class: {Class_id}, User: {User_id}")
        recording = True
        recording_frames = 0

# OpenGL and GLFW initialization
def init_window():
    if not glfw.init():
        raise Exception("GLFW cannot be initialized")

    # Create a landscape window (width > height)
    window = glfw.create_window(800, 600, "PointCloud", None, None)

    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")

    glfw.make_context_current(window)
    return window

def render_point_cloud(pointsData, window):
    width, height = glfw.get_framebuffer_size(window)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Set projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glViewport(0, 0, width, height)
    gluPerspective(60.0, float(width) / float(height), 0.01, 10.0)

    # Set view
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Camera setup
    eyeX = viewerState.lookatX
    eyeY = viewerState.lookatY
    eyeZ = viewerState.offset
    centerX = viewerState.lookatX
    centerY = viewerState.lookatY
    centerZ = 0.0
    gluLookAt(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, 0, 1, 0)

    # Rotate camera based on mouse
    glRotated(viewerState.pitch, 1, 0, 0)
    glRotated(viewerState.yaw, 0, 1, 0)
    glRotated(90, 0, 0, 1)

    glPointSize(2.5)  # Slightly larger points for better visibility
    glEnable(GL_DEPTH_TEST)

    glBegin(GL_POINTS)

    for point in pointsData:
        glColor3f(0.0, 1.0, 0.0)  # Green for hand points
        glVertex3f(point.worldX, point.worldY, point.worldZ)
    glEnd()

def detect_hands(rgb_image):
    input_image = rgb_image.copy()
    results = hands.process(input_image)

    
    hand_regions = []
    
    if results.multi_hand_landmarks:
        h, w, _ = rgb_image.shape
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(
                input_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Calculate the hand region (bounding box)
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Add padding to the bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w - 1, x_max + padding)
            y_max = min(h - 1, y_max + padding)
            
            # Draw rectangle around the hand
            cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            hand_regions.append((x_min, y_min, x_max, y_max))
    
    return input_image, hand_regions

def compute_cloud(depth_frame, rgbMat, hand_regions):

    height, width = depth_frame.shape[:2]
    rgb_height, rgb_width = rgbMat.shape[:2]

    cx, cy = width / 2, height / 2
    points_data = []

    # If no hands detected, return empty point cloud
    if not hand_regions:
        return []

    # Convert depth point to RGB coordinates (simple approximation)
    scale_x = rgb_width / width
    scale_y = rgb_height / height

    # Use a smaller step size for better performance
    step = 2
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Check if the point is in any hand region
            rgb_x = int(x * scale_x)
            rgb_y = int(y * scale_y)
            
            # For rotated images, transform coordinates
            # This assumes 90 degree counter-clockwise rotation
            if rgb_width != width or rgb_height != height:
                temp_x = rgb_x
                rgb_x = rgb_y
                rgb_y = rgb_width - temp_x - 1
            
            in_hand_region = False
            for x_min, y_min, x_max, y_max in hand_regions:
                if x_min <= rgb_x <= x_max and y_min <= rgb_y <= y_max:
                    in_hand_region = True
                    break
            
            if not in_hand_region:
                continue
                
            z = depth_frame[y, x]

            if z == 0 or z > 5000:
                continue
            
            z_m = z / scale
            if z_m > 0.5:  # Filter: hand closer than 50 cm
                continue

            # z_m = z / scale
            x_m = (x - cx) * z_m / fx
            y_m = (y - cy) * z_m / fy

            point = PointData()
            point.worldX = x_m
            point.worldY = -y_m
            point.worldZ = -z_m
            points_data.append(point)

    filtered_points = remove_outliers(points_data)
    return filtered_points

def compute_full_cloud(depth_frame, rgbMat):
    height, width = depth_frame.shape[:2]

    cx, cy = width / 2, height / 2
    points_data = []

    step = 2
    for y in range(0, height, step):
        for x in range(0, width, step):
            z = depth_frame[y, x]
            if z == 0 or z > 5000:
                continue

            z_m = z / scale
            x_m = (x - cx) * z_m / fx
            y_m = (y - cy) * z_m / fy

            point = PointData()
            point.worldX = x_m
            point.worldY = -y_m
            point.worldZ = -z_m
            points_data.append(point)

    return points_data

def remove_outliers(points, radius=0.03, min_neighbors=5):
    if not points:
        return []

    # Convert to NumPy array
    coords = np.array([[pt.worldX, pt.worldY, pt.worldZ] for pt in points])

    # Use NearestNeighbors to find local neighborhoods
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    neighbors = nbrs.radius_neighbors(coords, return_distance=False)

    # Keep only points with enough neighbors
    filtered_points = [
        pt for pt, neigh in zip(points, neighbors) if len(neigh) >= min_neighbors
    ]

    return filtered_points


# Define dataset root path at the top of your file
DATASET_ROOT = "dataset"  # You should change this to your desired path

def main():
    # Initialize the viewer state and GLFW window
    global viewerState, recording, recording_frames, Class_id, User_id
    viewerState = ViewerState()

    window = init_window()

    # Set up callbacks
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_key_callback(window, key_callback)
    
    # Set up window layout - using wide display windows
    cv2.namedWindow('RGB with Hand Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RGB with Hand Detection', 640, 480)
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth', 640, 480)
    cv2.namedWindow('IR', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('IR', 640, 480)
    
    # Position windows on the screen
    cv2.moveWindow('RGB with Hand Detection', 50, 30)
    cv2.moveWindow('Depth', 700, 30)
    cv2.moveWindow('IR', 50, 550)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        
        try:
            # Get synchronized depth frame
            depth_frame = np.frombuffer(
                depth_stream.read_frame().get_buffer_as_uint16(),
                dtype=np.uint16
            ).reshape((depth_stream.get_video_mode().resolutionY,
                      depth_stream.get_video_mode().resolutionX))
            depthMat1 = cv2.convertScaleAbs(depth_frame, alpha=255.0 / 1024.0)
            depthMat1 = cv2.applyColorMap(depthMat1, cv2.COLORMAP_JET)
            depthMat1 = cv2.rotate(depthMat1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            

            # Get synchronized RGB frame
            rgbFrame = color_stream.read_frame()
            rgbMat = np.frombuffer(rgbFrame.get_buffer_as_uint8(), dtype=np.uint8).reshape(
                rgbFrame.height, rgbFrame.width, 3)
            # rgbMat = cv2.cvtColor(rgbMat, cv2.COLOR_BGR2RGB)
            
            # Detect hands in the RGB image
            rgbMat_clean = rgbMat.copy()
            rgbMat_with_hands, hand_regions = detect_hands(rgbMat)
             
            
            # Compute the point cloud for hand regions only
            pointsData = compute_cloud(depth_frame, rgbMat, hand_regions)

            # IR frame processing
            irFrame = ir.read_frame()
            irMat = np.frombuffer(irFrame.get_buffer_as_uint16(), dtype=np.uint16).reshape(
                irFrame.height, irFrame.width, 1)
            irMat_display = cv2.convertScaleAbs(irMat, alpha=255.0 / 1024.0)
            
            # RECORDING BLOCK - Moved here after data is available
            if recording and recording_frames < max_recording_frames:
                save_dirs = create_data_dirs(DATASET_ROOT, Class_id, User_id)

                # Save RGB image
                rotated_rgb = cv2.rotate(rgbMat_clean, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rgb_path = os.path.join(save_dirs["rgb"], f"frame_{Class_id}_{User_id}_{recording_frames:03d}.png")
                cv2.imwrite(rgb_path, cv2.cvtColor(rotated_rgb, cv2.COLOR_RGB2BGR))  # Save as BGR for OpenCV

                # Save Depth image (raw)
                rotated_depth = cv2.rotate(depth_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                depth_raw_path = os.path.join(save_dirs["depth"], f"frame_{Class_id}_{User_id}_{recording_frames:03d}.png")
                # np.save(depth_raw_path.replace('.png', '.npy'), depth_frame)
                cv2.imwrite(depth_raw_path, depthMat1) 

                # Save Amplitude (IR) image
                rotated_amp = cv2.rotate(irMat_display , cv2.ROTATE_90_COUNTERCLOCKWISE)
                amp_path = os.path.join(save_dirs["amplitude"], f"frame_{Class_id}_{User_id}_{recording_frames:03d}.png")
                cv2.imwrite(amp_path, rotated_amp)

                # Save point cloud data
                pc_path = os.path.join(save_dirs["point_cloud"], f"frame_{Class_id}_{User_id}_{recording_frames:03d}.ply")
                with open(pc_path, 'w') as f:
                    for pt in pointsData:
                        X_rot = -pt.worldY
                        Y_rot = pt.worldX
                        Z_rot = pt.worldZ
                        f.write(f"{X_rot} {Y_rot} {Z_rot}\n")

                # === Save Full Point Cloud ===
                pointsData_full = compute_full_cloud(depth_frame, rgbMat)

                full_pc_path = os.path.join(save_dirs["full_point_cloud"], f"frame_{Class_id}_{User_id}_{recording_frames:03d}.ply")
                with open(full_pc_path, 'w') as f:
                    for pt in pointsData_full:
                        X_rot = -pt.worldY
                        Y_rot = pt.worldX
                        Z_rot = pt.worldZ
                        f.write(f"{X_rot} {Y_rot} {Z_rot}\n")


                recording_frames += 1

                if recording_frames == max_recording_frames:
                    recording = False
                    print("Recording finished.")

            # Render point cloud
            render_point_cloud(pointsData, window)

            # Display RGB image with hand landmarks - landscape orientation
            rgbMat_with_hands = cv2.rotate(rgbMat_with_hands, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow('RGB with Hand Detection', rgbMat_with_hands)

            # Process depth frame (same as the one used for point cloud) - landscape orientation
            depthMat = cv2.convertScaleAbs(depth_frame, alpha=255.0 / 1024.0)
            depthMat = cv2.applyColorMap(depthMat, cv2.COLORMAP_JET)
            depthMat = cv2.rotate(depthMat, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Draw boxes around hands on the depth image
            for x_min, y_min, x_max, y_max in hand_regions:
                try:
                    # Need to transform the coordinates for rotated image
                    h, w = rgbMat.shape[:2]
                    # Adjust coordinates for rotation and possible scaling between RGB and depth
                    depth_h, depth_w = depthMat.shape[:2]
                    scale_x = depth_w / h
                    scale_y = depth_h / w
                    
                    # For rotated coordinates (90° counterclockwise)
                    new_x_min = int(y_min * scale_y)
                    new_y_min = int((w - x_max - 1) * scale_x)
                    new_x_max = int(y_max * scale_y)
                    new_y_max = int((w - x_min - 1) * scale_x)

                    # 💡 Shift box 40 pixels left
                    shift = 40
                    new_x_min = max(0, new_x_min - shift)
                    new_x_max = max(0, new_x_max - shift)

                    new_x_min = min(depth_w - 1, new_x_min)
                    new_x_max = min(depth_w - 1, new_x_max)
                    new_y_min = max(0, min(depth_h - 1, new_y_min))
                    new_y_max = max(0, min(depth_h - 1, new_y_max))
                    
                    
                    # Ensure min values are less than max values
                    if new_x_min >= new_x_max or new_y_min >= new_y_max:
                        continue
                    
                    cv2.rectangle(depthMat, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 255, 255), 2)
                except Exception as e:
                    print(f"Error drawing depth rectangle: {e}")
            
            cv2.imshow('Depth', depthMat)

            # IR frame processing - landscape orientation
            irFrame = ir.read_frame()
            irMat = np.frombuffer(irFrame.get_buffer_as_uint16(), dtype=np.uint16).reshape(
                irFrame.height, irFrame.width, 1)
            irMat = cv2.convertScaleAbs(irMat, alpha=255.0 / 1024.0)
            irMat = cv2.cvtColor(irMat, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing colored boxes
            irMat = cv2.rotate(irMat, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Draw boxes around hands on the IR image (similar transformation as for depth)
            for x_min, y_min, x_max, y_max in hand_regions:
                try:
                    h, w = rgbMat.shape[:2]
                    ir_h, ir_w = irMat.shape[:2]
                    scale_x = ir_w / h
                    scale_y = ir_h / w

                    new_x_min = int(y_min * scale_y)
                    new_y_min = int((w - x_max - 1) * scale_x)
                    new_x_max = int(y_max * scale_y)
                    new_y_max = int((w - x_min - 1) * scale_x)

                    # 💡 Shift box 40 pixels left
                    shift = 40
                    new_x_min = max(0, new_x_min - shift)
                    new_x_max = max(0, new_x_max - shift)

                    # Clamp to image size
                    new_x_min = min(ir_w - 1, new_x_min)
                    new_x_max = min(ir_w - 1, new_x_max)
                    new_y_min = max(0, min(ir_h - 1, new_y_min))
                    new_y_max = max(0, min(ir_h - 1, new_y_max))

                    # Ensure min < max
                    if new_x_min >= new_x_max or new_y_min >= new_y_max:
                        continue

                    cv2.rectangle(irMat, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 255, 255), 2)
                except Exception as e:
                    print(f"Error drawing IR rectangle: {e}")

                
            cv2.imshow('IR', irMat)

        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Swap buffers to display the rendered frame
        glfw.swap_buffers(window)

    # Clean up
    try:
        hands.close()
    except:
        pass
        
    glfw.terminate()
    depth_stream.stop()
    color_stream.stop()
    ir.stop()
    openni2.unload()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()