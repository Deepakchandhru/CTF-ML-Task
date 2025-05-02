import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk

class VideoProcessor:
    def __init__(self, input_device=0, output_video_path="virtual_bg.avi",
                 fps=30, frame_width=640, frame_height=480, 
                 record_output_path="recorded_video.avi",
                 green_bg_output="green_screen.avi",
                 bg_image_path="background.jpg"):
        
        self.input_device = input_device
        self.output_video_path = output_video_path
        self.record_output_path = record_output_path
        self.green_bg_output = green_bg_output
        self.bg_image_path = bg_image_path

        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.running = False

        # Initialize Mediapipe Selfie Segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        # Video writers
        self.out_recorded = None  
        self.out_green_bg = None  
        self.out_virtual_bg = None  

        self.start_time = None  
        self.initial_delay = 2  

        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()

        # Load Background Image
        self.bg_image = cv2.imread(self.bg_image_path)
        if self.bg_image is None:
            print("Warning: Background image not found! Using default solid color.")
            self.bg_image = np.full((self.frame_height, self.frame_width, 3), (50, 50, 200), dtype=np.uint8)  # Default BG (blue)

    def apply_green_screen(self, frame):
        """Applies green screen effect using Mediapipe Selfie Segmentation"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(frame_rgb)
        mask = results.segmentation_mask

        condition = np.stack((mask,) * 3, axis=-1) > 0.5  # More precise mask
        green_bg = np.full(frame.shape, (0, 255, 0), dtype=np.uint8)
        green_screen_frame = np.where(condition, frame, green_bg)

        return green_screen_frame, mask

    def apply_virtual_bg(self, frame, mask):
        """Replaces background with custom image"""
        bg_resized = cv2.resize(self.bg_image, (self.frame_width, self.frame_height))
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        virtual_bg_frame = np.where(condition, frame, bg_resized)

        return virtual_bg_frame

    def record_video(self):
        """Captures video, applies green screen, and replaces background in real time."""
        cap = cv2.VideoCapture(self.input_device)
        if not cap.isOpened():
            print(f"Error: Could not open video capture device {self.input_device}. Exiting.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("Recording... Press 'q' to stop.")
        self.start_time = time.time()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        self.out_recorded = cv2.VideoWriter(self.record_output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.out_green_bg = cv2.VideoWriter(self.green_bg_output, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.out_virtual_bg = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame. Stopping recording.")
                break

            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.initial_delay:
                if self.out_virtual_bg is None:
                    print("Processing Started")

            # Apply Green Screen Effect
            green_screen_frame, mask = self.apply_green_screen(frame)

            # Apply Virtual Background
            virtual_bg_frame = self.apply_virtual_bg(frame, mask)

            # Stack all 3 frames side by side in a single window
            combined_display = np.hstack((frame, green_screen_frame, virtual_bg_frame))

            # Resize the final display window to fit the screen properly
            display_resized = cv2.resize(combined_display, (self.screen_width, self.screen_height))

            # Show the combined display
            cv2.imshow("Live | Green Screen | Virtual Background", display_resized)

            # Save all three videos
            self.out_recorded.write(frame)
            self.out_green_bg.write(green_screen_frame)
            self.out_virtual_bg.write(virtual_bg_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if self.out_recorded:
            self.out_recorded.release()
        if self.out_green_bg:
            self.out_green_bg.release()
        if self.out_virtual_bg:
            self.out_virtual_bg.release()
        
        cv2.destroyAllWindows()
        print("Recording stopped. Videos saved.")

    def start(self):
        """Starts the recording and processing."""
        self.running = True
        self.record_video()

    def stop(self):
        """Stops the recording process."""
        self.running = False
        print("Process stopped")


if __name__ == "__main__":
    processor = VideoProcessor(bg_image_path="background.jpg")
    processor.start()

