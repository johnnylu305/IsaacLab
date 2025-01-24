import cv2
import os

def merge_images_to_video(folder_path, output_video, fps=30):
    """
    Merge PNG images starting with 'vis' in a folder into a video.

    Parameters:
        folder_path (str): Path to the folder containing PNG images.
        output_video (str): Path to save the output video file.
        fps (int): Frames per second for the video. Default is 30.

    Returns:
        None
    """
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if f.startswith("depth") and f.endswith(".png")]

    # Sort the files to ensure the order is correct
    files.sort()
    
    # Check if there are any images
    if not files:
        print("No images starting with 'vis' found in the folder.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(folder_path, files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Loop through the images and write them to the video
    for file_name in files:
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Unable to read image {image_path}. Skipping.")
            continue

        video_writer.write(image)

    # Release the VideoWriter object
    video_writer.release()
    print(f"Video saved as {output_video}")

# Example usage
if __name__ == "__main__":
    
    folder_path = "/projects/MAD3D/Zhuoli/IsaacLab/logs/sb3/MAD3D-v0/objaverse_data"  # Replace with the path to your folder
    for folder in os.listdir(folder_path):
        if "ad4fe73f232840419e10a2b9e52cc729" not in folder:
            continue

        if os.path.isdir(os.path.join(folder_path, folder)): 
        #print(folder)
            output_video = os.path.join(folder_path,f"{folder}_depth_video.mp4")  # Replace with the desired output video name

            merge_images_to_video(os.path.join(folder_path, folder), output_video, fps=0.6)
        

