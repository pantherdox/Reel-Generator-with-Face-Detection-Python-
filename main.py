import os
import cv2
import dlib
import logging
import time
from moviepy.editor import VideoFileClip

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load face detector
detector = dlib.get_frontal_face_detector()

def process_frame(frame, prev_face, target_face, crop_width, crop_height, frame_count, switch_duration, last_switch_frame):
    faces = detector(frame)  # Detect faces in the current frame
    
    if len(faces) > 0:
        current_faces = []
        for face in faces:
            face_center_x = (face.left() + face.right()) // 2
            face_center_y = (face.top() + face.bottom()) // 2
            face_width = face.width()
            face_height = face.height()
            face_area = face_width * face_height
            current_faces.append((face_area, face_center_x, face_center_y))

        current_faces.sort(key=lambda f: f[0], reverse=True)
        most_prominent_face = current_faces[0] if len(current_faces) > 0 else target_face

        if target_face is None or frame_count - last_switch_frame >= switch_duration:
            target_face = most_prominent_face
            last_switch_frame = frame_count

        # Smoothly interpolate between previous and new target face position over several frames
        if prev_face is not None:
            blend_factor = min(1.0, (frame_count - last_switch_frame) / switch_duration)
            face_center_x = int(prev_face[1] * (1 - blend_factor) + target_face[1] * blend_factor)
            face_center_y = int(prev_face[2] * (1 - blend_factor) + target_face[2] * blend_factor)
        else:
            face_center_x, face_center_y = target_face[1], target_face[2]
        
        prev_face = (target_face[0], face_center_x, face_center_y)

    else:
        logging.warning("No faces detected in the current frame.")
        if prev_face is not None:
            face_center_x, face_center_y = prev_face[1], prev_face[2]
        else:
            face_center_x, face_center_y = crop_width // 2, crop_height // 2

    crop_x1 = max(0, face_center_x - crop_width // 2)
    crop_y1 = max(0, face_center_y - crop_height // 2)
    crop_x2 = min(frame.shape[1], crop_x1 + crop_width)
    crop_y2 = min(frame.shape[0], crop_y1 + crop_height)

    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    return cropped_frame, prev_face, target_face, last_switch_frame

def center_active_speaker(input_video_path, temp_video_path, crop_ratio=(9, 16), tolerance=0.8):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        logging.error(f"Error: Could not open video at {input_video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"Video loaded: {input_video_path} (Width: {width}, Height: {height}, FPS: {fps})")

    crop_width = int(height * crop_ratio[0] / crop_ratio[1])
    crop_height = height
    if crop_width > width:
        crop_width = width
        crop_height = int(width * crop_ratio[1] / crop_ratio[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (crop_width, crop_height))

    prev_face = None
    target_face = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    switch_duration = 3 * fps  # Duration in frames for smooth transition
    last_switch_frame = 0
    
    logging.info(f"Starting video processing: {total_frames} total frames.")

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        cropped_frame, prev_face, target_face, last_switch_frame = process_frame(
            frame, prev_face, target_face, crop_width, crop_height, frame_count, switch_duration, last_switch_frame
        )
        processing_time = time.time() - start_time

        out.write(cropped_frame)

        if frame_count % 10 == 0:
            logging.info(f"Processed frame {frame_count}/{total_frames} - Processing time: {processing_time:.2f} seconds")

    cap.release()
    out.release()
    logging.info(f"Processed video saved at {temp_video_path}")

def process_videos(input_folder, output_folder, video_filename):
    input_video_path = os.path.join(input_folder, video_filename)
    temp_video_name = f"temp_{video_filename}"
    temp_video_path = os.path.join(output_folder, temp_video_name)
    output_video_path = os.path.join(output_folder, f"processed_{video_filename}")

    center_active_speaker(input_video_path, temp_video_path)

    logging.info("Adding audio to the processed video...")
    try:
        video_clip = VideoFileClip(input_video_path)
        processed_clip = VideoFileClip(temp_video_path)

        processed_clip = processed_clip.set_audio(video_clip.audio)
        processed_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

        logging.info(f"Final video with audio saved at {output_video_path}")

    except Exception as e:
        logging.error(f"Error while adding audio: {e}")

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        video_clip.close()
        processed_clip.close()

if __name__ == "__main__":
    input_folder = 'input_videos'
    output_folder = 'output_videos'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_filename = 'input_video.mp4'  # Replace with the actual video filename

    process_videos(input_folder, output_folder, video_filename)

