
# Prototype Workflow for Semi-Automatic Processing of Demonstrative Pronouns and Pointing Gestures

The current prototype is designed to analyse the temporal alignment of spoken demonstrative pronouns and pointing gestures in video recordings. The workflow integrates computer vision (via MediaPipe) for gesture detection and audio processing (via a language-specific speech recognition model) to extract relevant linguistic features. The pipeline comprises multiple scripts that handle different aspects of the processing, culminating in an enriched video with annotations of detected events. 

For reference, this repository includes an [*ELAN*](/ELAN) folder containing output from a traditional annotation process using the [*ELAN* program](https://archive.mpi.nl/tla/elan). Ultimately, the performance of the semi-automated prototype must be validated against these ELAN-based annotations.

## Running the Program

The system requires primary data in the form of video and corresponding audio files, which should be placed in `mnt/primary data`. They video-audio pairs should be named in the same way (e.g., `1.mp4` and `1.wav`). The video should feature a person in a medium or medium close-up shot. 

```
python main.py --audio_folder "mnt/primary data/audio" \
               --video_folder "mnt/primary data/video" \
               --demonstratives "der,die,das,den,dem,denen,dessen,deren,dieser,diese,dieses,diesen,diesem" \
               --model "mnt/primary data/vosk-model-de-0.21" \
               --output "mnt/output" \
               --max_time_diff 800
```


## Pipeline Overview

### 1. Audio Transcription & Word Onset Extraction (audio_processing.py)

- Speech is transcribed using a Vosk language model. 

- The script extracts demonstrative pronouns (based on a predefined list of such pronouns) along with their onset times.

- A plain text transcript and a WebVTT subtitle file are generated.

### 2. Gesture Detection (video_processing.py)

- MediaPipe’s hand landmarks estimation is used to detect the apexes of pointing gestures (see [demonstration](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) of the hand landmarks detection). Specifically, pointing gestures are identified as the moment at which the wrist (i.e., hand landmark `0`) and the tip of the index finger (i.e., hand landmark `8`) are most distant from each other. 

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

Currently, the system overidentifies demonstrative pronouns by including any homographs from the list of demonstrative pronouns. For example, in languages such as English, French and German, many definite articles are mistakenly included because they share the same form as demonstrative pronouns. This issue could be addressed by replacing the current fuzzy word list with a more precise list, where each pronoun is contextualised by its preceding and subsequent words.

### 2. Enhancing the accuracy of pointing gesture detection

Currently, the program underidentifies pointing gestures. Improvements to the implementation of MediaPipe’s detection or additional filtering techniques (e.g., movement velocity thresholds) are needed.
This prototype serves as a foundational step towards a fully automated analysis of deictic communication, combining speech and gesture processing for richer linguistic insights.







