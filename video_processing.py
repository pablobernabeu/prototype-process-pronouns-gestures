import cv2
import mediapipe as mp
import numpy as np
import logging

def detect_multiple_gesture_apexes(video_path, threshold=0.02, min_frames=2, min_gap_frames=5):
    '''
    Detects multiple gesture apexes from a video file using MediaPipe.
    
    Arguments:
        video_path (str): Path to the video file.
        threshold (float): Minimum extension threshold to detect an active gesture.
        min_frames (int): Minimum consecutive frames to consider a gesture valid.
        min_gap_frames (int): Minimum frames between separate gestures to prevent merging.
        
    Returns:
        list of float: List of detected gesture apex times in milliseconds.
    '''
    try:
        # Initialise MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, 
                               max_num_hands=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.3)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return []

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        hand_extension = []  # Stores (frame_idx, max_extension_value)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            frame_max_extension = 0.0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
                    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
                    distance = np.linalg.norm(index_tip - wrist)
                    frame_max_extension = max(frame_max_extension, distance)
                
            hand_extension.append((frame_count, frame_max_extension if frame_max_extension > 0 else 0.0))
        
        cap.release()
        cv2.destroyAllWindows()

        if not hand_extension:
            logging.warning('No frames processed or no hand gestures detected.')
            return []

        # Detect gestures with gaps accounted for
        apex_frames = []
        current_gesture_frames = []
        last_apex_frame = -min_gap_frames  # Ensures first gesture is considered

        for (f_idx, extension) in hand_extension:
            if extension > threshold:
                current_gesture_frames.append((f_idx, extension))
            else:
                if len(current_gesture_frames) >= min_frames:
                    apex_frame, _ = max(current_gesture_frames, key=lambda x: x[1])
                    if apex_frame - last_apex_frame > min_gap_frames:
                        apex_frames.append(apex_frame)
                        last_apex_frame = apex_frame
                current_gesture_frames = []
        
        # Edge case: if the last frames in the video formed a gesture
        if len(current_gesture_frames) >= min_frames:
            apex_frame, _ = max(current_gesture_frames, key=lambda x: x[1])
            if apex_frame - last_apex_frame > min_gap_frames:
                apex_frames.append(apex_frame)

        # Convert frame indices to times (in milliseconds)
        apex_times = [(frame_idx / frame_rate) * 1000 for frame_idx in apex_frames]
        
        logging.info(f"Detected {len(apex_times)} gesture apex(es).")
        return apex_times

    except Exception as e:
        logging.error(f"Error in detecting multiple gesture apexes: {e}")
        return []
