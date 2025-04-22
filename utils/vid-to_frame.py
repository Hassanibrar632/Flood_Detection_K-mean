import cv2
import os

# extract the Video frames and save them in a folder
def extract_frames(video_path, output_folder='oputput_frames'):
    """
    Extract frames from a video file and save them as images in a specified folder.
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where extracted frames will be saved.

    Returns:
        success (bool): True if frames were extracted successfully, False otherwise.
        output_path (str): Path to the folder where frames are saved.
    """
    output_path = os.path.join(output_folder, os.path.basename(video_path).split(".")[0])

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output folder: {output_path}")

    # Initialize success flag
    success = True

    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        return False, None
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error opening video file: {video_path}")
        return False, None

    # Read and save frames
    rec, frame = video.read()
    saved_count = 0

    # Loop through the video frames and save them as images
    while rec:
        frame_name = os.path.join(output_path, f"frame_{saved_count}.jpg")
        cv2.imwrite(frame_name, frame)
        saved_count += 1
        rec, frame = video.read()

    # Release the video capture object and return success
    video.release()
    return success, output_path

if __name__ == "__main__":
    # Example usage
    video_path = r"Video-Data\video.mp4"  # Path to your video file
    output_folder = r"./EXTRACTED_FRAMES"  # Output folder for extracted frames

    # Extract frames from the video
    extract_frames(video_path, output_folder)