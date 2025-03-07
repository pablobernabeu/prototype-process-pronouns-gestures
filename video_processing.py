# video_processing.py

import cv2
import mediapipe as mp
import numpy as np
import logging

def detect_multiple_gesture_apexes(video_path, threshold=0.1, min_frames=5):
    '''
    Processes the video file using MediaPipe to detect hand gestures and calculates
    multiple apexes of the pointing gesture. The apex is defined as the local maximum
    of index-finger extension (distance between the wrist and the index finger tip).
    
    Arguments:
        video_path (str): Path to the video file.
        threshold (float): Extension threshold for considering a hand gesture as "active".
                           Only frames with extension > threshold are treated as part
                           of a gesture.
        min_frames (int): Minimum number of consecutive frames above the threshold
                          required to count as a valid gesture.
                          
    Returns:
        list of float: A list of gesture apex times (in ms).
    '''
    try:
        # Initialise MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,
                               max_num_hands=2,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return []

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        # This will store (frame_idx, extension_value) for each frame
        hand_extension = []

        # Read frames and calculate extension
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # In case multiple hands are detected, pick the maximum extension
                # across all detected hands for that frame
                frame_max_extension = 0.0
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
                    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
                    distance = np.linalg.norm(index_tip - wrist)
                    frame_max_extension = max(frame_max_extension, distance)
                
                hand_extension.append((frame_count, frame_max_extension))
            else:
                # No hand detected in this frame, treat extension as 0
                hand_extension.append((frame_count, 0.0))
        
        cap.release()
        cv2.destroyAllWindows()

        if not hand_extension:
            logging.warning('No frames processed or no hand gestures detected.')
            return []

        # Now, detect multiple gestures by looking for consecutive frames above threshold
        apex_frames = []
        current_gesture_frames = []

        for (f_idx, extension) in hand_extension:
            if extension > threshold:
                # This frame is part of an 'active' gesture
                current_gesture_frames.append((f_idx, extension))
            else:
                # The gesture ended; if it meets min_frames, find local apex
                if len(current_gesture_frames) >= min_frames:
                    apex_frame, _ = max(current_gesture_frames, key=lambda x: x[1])
                    apex_frames.append(apex_frame)
                current_gesture_frames = []
        
        # Edge case: if the last frames in the video formed a gesture
        if len(current_gesture_frames) >= min_frames:
            apex_frame, _ = max(current_gesture_frames, key=lambda x: x[1])
            apex_frames.append(apex_frame)

        # Convert frame indices to times (in milliseconds)
        apex_times = [(frame_idx / frame_rate) * 1000 for frame_idx in apex_frames]
        
        logging.info(f"Detected {len(apex_times)} gesture apex(es).")
        return apex_times
    
    except Exception as e:
        logging.error(f"Error in detecting multiple gesture apexes: {e}")
        return []
