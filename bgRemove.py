import cv2
import numpy as np
import mediapipe as mp

def extract_person_from_video_mediapipe(input_video_path, output_video_path):
    """
    Extracts a person from a video using Mediapipe's selfie segmentation model and replaces the background with green.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
    """
    print("Opening video file...")
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'H264' for mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1) # 0 (general) or 1 (landscape)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Prepare the frame for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(frame_rgb)
        mask = results.segmentation_mask

        # Condition to separate foreground (person) and background.
        condition = np.stack((mask,) * 3, axis=-1) > 0.6  # Adjust threshold as needed

        # Create a green background frame (BGR format).
        bg_green = np.full(frame.shape, (0, 255, 0), dtype=np.uint8)

        # Combine the original frame with the green background using the mask.
        output_image = np.where(condition, frame, bg_green)

        # Write the frame to the output video
        out.write(output_image)

    print("Processing complete. Releasing resources...")
    cap.release()
    out.release()
    selfie_segmentation.close()
    print("Video processing finished.")

if __name__ == "__main__":
    input_video = "background/bg1.mp4"  # Replace with your input video path
    output_video = "background/mediapipe_green.mp4"
    print("Starting person extraction from video using Mediapipe...")
    extract_person_from_video_mediapipe(input_video, output_video)
    print(f"Output video saved to {output_video}")

