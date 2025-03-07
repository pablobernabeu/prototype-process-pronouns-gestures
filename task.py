import json
import wave
import cv2
import mediapipe as mp
import numpy as np
from vosk import Model, KaldiRecognizer
import matplotlib.pyplot as plt
import pandas as pd

def transcribe_audio(audio_path, model_path):
    """
    Transcribes the audio file using Vosk and returns a list of recognition results.
    """
    try:
        # Initialise the Vosk model
        model = Model(model_path)
        wf = wave.open(audio_path, 'rb')
        # Ensure the audio file is in the correct format (mono, 16-bit PCM)
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != 'NONE':
            raise ValueError('Audio file must be a WAV file in mono PCM format.')
        
        recogniser = KaldiRecognizer(model, wf.getframerate())
        recogniser.SetWords(True)
        results = []
        
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recogniser.AcceptWaveform(data):
                result = json.loads(recogniser.Result())
                results.append(result)
            else:
                partial_result = json.loads(recogniser.PartialResult())
                if 'result' in partial_result:
                    results.append(partial_result)  # Store partial results with word timestamps
        
        # Append any final results
        final_result = json.loads(recogniser.FinalResult())
        results.append(final_result)
        print("Transcription results:", results)  # Debugging statement
        return results
    except Exception as e:
        print(f"Error in transcribing audio: {e}")
        return []

def extract_demonstrative_onsets(results, demonstratives=['der', 'die', 'da', 'das', 'den', 'dem', 'denen', 'dessen', 'deren', 'dieser', 'diese', 'dieses', 'diesen', 'diesem']):
    """
    Extracts the onset times (in seconds) of demonstrative pronouns from the recogniser results.
    """
    onsets = []
    for segment in results:
        print(f"Segment: {segment}")  # Debugging statement
        if 'result' in segment and isinstance(segment['result'], list):
            for word_info in segment['result']:
                if 'word' in word_info and 'start' in word_info:
                    word = word_info['word'].lower()
                    print(f"Checking word: {word}")  # Debugging statement
                    if word in demonstratives:
                        onsets.append(word_info['start'])
    print("Extracted onsets:", onsets)  # Debugging statement
    return onsets

def detect_gesture_apex(video_path):
    """
    Processes the video file using MediaPipe to detect hand gestures and calculates the apex of the pointing gesture.
    The apex is operationally defined here as the frame with the maximum extension of the index finger (distance between
    the wrist and the index finger tip). Returns a list of apex times in seconds.
    """
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,
                               max_num_hands=2,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(video_path)
        gesture_apex_times = []
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        hand_extension = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Convert frame colour space from BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Compute distance between the wrist (landmark 0) and the tip of the index finger (landmark 8)
                    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
                    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
                    distance = np.linalg.norm(index_tip - wrist)
                    hand_extension.append((frame_count, distance))
                    # Optionally, draw the hand landmarks for debugging:
                    # mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cap.release()
        cv2.destroyAllWindows()

        if not hand_extension:
            print('No hand gestures detected.')
            return gesture_apex_times

        # Determine the apex as the frame with the maximum hand extension
        apex_frame, max_extension = max(hand_extension, key=lambda x: x[1])
        apex_time = apex_frame / frame_rate
        gesture_apex_times.append(apex_time)
        return gesture_apex_times
    except Exception as e:
        print(f"Error in detecting gesture apex: {e}")
        return []

def calculate_alignment(demo_onsets, gesture_apex_times):
    """
    Calculates the temporal differences between each demonstrative onset and the closest gesture apex.
    Returns a list of differences in seconds.
    """
    alignments = []
    for onset in demo_onsets:
        differences = [abs(onset - apex) for apex in gesture_apex_times]
        if differences:
            alignments.append(min(differences))
        else:
            alignments.append(None)
    return alignments

def main():
    try:
        # Define file paths and the path to the German Vosk model.
        audio_path = 'mnt/primary data/audio.wav'
        video_path = 'mnt/primary data/video.mp4'
        german_model_path = 'mnt/primary data/vosk-model-de-0.21'  # Update with the correct path to your Vosk German model

        # Step 1: Transcribe audio and extract demonstrative onset times.
        print('Transcribing audio...')
        transcription_results = transcribe_audio(audio_path, german_model_path)
        demo_onsets = extract_demonstrative_onsets(transcription_results)
        print('Demonstrative onset times (seconds):', demo_onsets)

        # Step 2: Process video to detect the gesture apex.
        print('Processing video to detect gesture apex...')
        gesture_apex_times = detect_gesture_apex(video_path)
        print('Gesture apex times (seconds):', gesture_apex_times)

        # Step 3: Calculate alignment between speech and gesture.
        alignment_differences = calculate_alignment(demo_onsets, gesture_apex_times)
        print('Temporal differences between demonstrative onsets and gesture apexes (seconds):', alignment_differences)
        
        # Create a DataFrame for clarity
        df = pd.DataFrame({
            'demonstrative_onset': demo_onsets,
            'alignment_difference': alignment_differences
        })

        # Plot a histogram of the temporal differences
        plt.figure(figsize=(10, 6))
        plt.hist(df['alignment_difference'], bins=10, edgecolor='black')
        plt.title('Distribution of Temporal Differences')
        plt.xlabel('Temporal Difference (seconds)')
        plt.ylabel('Frequency')
        plt.show()

        # Alternatively, plot a scatter plot comparing demonstrative onsets with alignment differences
        plt.figure(figsize=(10, 6))
        plt.scatter(df['demonstrative_onset'], df['alignment_difference'], c='blue')
        plt.title('Demonstrative Onset vs Temporal Difference')
        plt.xlabel('Demonstrative Onset (seconds)')
        plt.ylabel('Temporal Difference (seconds)')
        plt.show()
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == '__main__':
    main()