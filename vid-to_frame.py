import cv2
import os

def extract_frames(video_path, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    success, frame = video.read()  # Read the first frame
    saved_count = 0  # Counter for saved frames

    while success:
        frame_name = os.path.join(output_folder, f"frame_{saved_count}.jpg")
        cv2.imwrite(frame_name, frame)
        saved_count += 1
        success, frame = video.read()  # Read the next frame

    video.release()  # Release the video resource
    print(f"Extracted {saved_count} frames from {video_path} and saved to {output_folder}")

# Set paths
output_folder = r"./EXTRACTED_FRAMES" # Output folder
extract_frames("video.mp4", output_folder)