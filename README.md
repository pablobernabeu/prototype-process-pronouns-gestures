
# Prototype Workflow for Semi-Automatic Processing of Demonstrative Pronouns and Pointing Gestures

The current prototype is designed to analyse the temporal alignment of spoken demonstrative pronouns and pointing gestures in video recordings. The workflow integrates computer vision (via MediaPipe) for gesture detection and audio processing (via a language-specific speech recognition model) to extract relevant linguistic features. The pipeline comprises multiple scripts that handle different aspects of the processing, culminating in an enriched video with annotations of detected events.

## Pipeline Overview

### 1. Audio Transcription & Word Onset Extraction (audio_processing.py)

- Speech is transcribed using a Vosk language model.

- The script extracts demonstrative pronouns along with their onset times.

- A plain text transcript and a WebVTT subtitle file are generated.

### 2. Gesture Detection (video_processing.py)

- MediaPipe’s hand landmarks estimation is used to detect gesture the apexes of pointing gestures (see [demonstration](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) of the hand landmarks detection).

### 3. Alignment Analysis (alignment_analysis.py)

- Onset times of demonstrative pronouns are compared with detected gesture apexes.

- The script calculates the temporal difference between the two.

- Outputs a CSV file containing word-gesture alignment data.

- Generates visualisations (histogram and scatter plot) to illustrate alignment patterns.

### 4. Video Processing & Annotation (video_editing.py)

- The original audio is merged into the video.

- Subtitles displaying the transcribed speech are added.

- Gesture peaks are highlighted in the video.

### 5. Automated Execution (main.py)

- Coordinates all steps of the pipeline.

- Processes multiple audio–video file pairs from a specified directory.

- Ensures results are systematically stored in the output directory.


## Outstanding Tasks

### 1. Filtering out definite articles from the extracted demonstrative pronouns

Currently, the system captures all words in the transcriptions, including definite articles (e.g., the), which should be excluded from the demonstrative pronoun dataset.

### 2. Enhancing the accuracy of pointing gesture detection

Some gestures are missed or falsely identified. Improvements to MediaPipe’s detection logic or additional filtering techniques (e.g., movement velocity thresholds) are needed.
This prototype serves as a foundational step towards a fully automated analysis of deictic communication, combining speech and gesture processing for richer linguistic insights.







