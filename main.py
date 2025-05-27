
import argparse
import os
import webvtt
import glob
import logging
import pandas as pd
import matplotlib.pyplot as plt

from audio_processing import transcribe_audio
from video_processing import detect_multiple_gesture_apices
from alignment_analysis import extract_word_of_interest_onsets, calculate_alignment
from video_editing import merge_audio_video, add_captions, signal_gesture_peaks

def process_pair(audio_path, video_path, model_path, output_dir, max_time_diff, words_of_interest):
    '''
    Processes a matching audio and video file pair.

    Steps:
    1. Transcribe the audio and export plain text and VTT subtitles.
    2. Extract onsets of target words from the transcription.
    3. Detect gesture apices using computer vision.
    4. Align gestures with words of interest within a defined temporal window.
    5. Extract surrounding word context from transcript for each word of interest.
    6. Plot the temporal alignment differences with annotated scatterplot.
    7. Merge audio and video into a single output file.
    8. Add subtitles as captions to the video.
    9. Annotate gesture apices visually in the video.
    '''

    # Step 1: Transcribe audio and export plain text
    transcription_results = transcribe_audio(audio_path, model_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}_transcription.txt")

    all_text_segments = [segment.get('text', '') for segment in transcription_results if segment.get('text')]
    full_transcription = "\n".join(all_text_segments)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_transcription)
    print(f"Plain text transcription saved to {txt_path}")

    # Export WebVTT subtitle file
    def sec_to_timestamp(seconds):
        import math
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        msecs = int(round((secs - int(secs)) * 1000))
        secs_int = int(secs)
        return f"{hours:02d}:{minutes:02d}:{secs_int:02d}.{msecs:03d}"

    vtt = webvtt.WebVTT()
    for segment in transcription_results:
        if 'result' in segment and segment['result']:
            start_time = segment['result'][0].get('start', 0)
            end_time = segment['result'][-1].get('end', start_time)
            text = segment.get('text', '')
            if text:
                caption = webvtt.Caption(start=sec_to_timestamp(start_time), end=sec_to_timestamp(end_time), text=text)
                vtt.captions.append(caption)

    vtt_path = os.path.join(output_dir, f"{base_name}.vtt")
    vtt.save(vtt_path)
    print(f"WebVTT captions saved to {vtt_path}")

    # Step 2: Extract target words and their onset times
    word_onsets = extract_word_of_interest_onsets(transcription_results, words_of_interest)

    # Step 3: Detect gesture apex frames
    gesture_apex_times = detect_multiple_gesture_apices(video_path)

    # Step 4 & 5: Align gestures to words and extract local context
    df_rows = []
    used_apices = set()

    transcript_chunks = [
        (segment['result'][0]['start'], segment['result'][-1]['end'], segment['text'])
        for segment in transcription_results if 'result' in segment and segment['result']
    ]

    for word_of_interest, onset_time in word_onsets:
        matched_context = None
        for start_time, end_time, sentence in transcript_chunks:
            if start_time <= onset_time <= end_time:
                words = sentence.split()
                if word_of_interest in words:
                    index = words.index(word_of_interest)
                    prev_word = words[index - 1] if index > 0 else ""
                    next_word = words[index + 1] if index < len(words) - 1 else ""
                    matched_context = f"{prev_word} {word_of_interest} {next_word}".strip()
                    break
        if not matched_context:
            matched_context = word_of_interest

        available_apices = [
            apex for apex in gesture_apex_times
            if apex not in used_apices and abs(onset_time - apex) <= max_time_diff
        ]

        if available_apices:
            nearest_apex = min(available_apices, key=lambda apex: abs(onset_time - apex))
            used_apices.add(nearest_apex)
            alignment_diff = onset_time - nearest_apex
        else:
            nearest_apex = None
            alignment_diff = None

        df_rows.append({
            'word_of_interest': word_of_interest,
            'word_of_interest_onset': onset_time,
            'gesture_apex': nearest_apex,
            'alignment_difference': alignment_diff,
            'word_of_interest_context': matched_context
        })

    df = pd.DataFrame(df_rows)

    # Export alignment data
    csv_path = os.path.join(output_dir, f"{base_name}_alignment.csv")
    df.to_csv(csv_path, index=False)
    print(f"Alignment data saved to {csv_path}")

    # Step 6: Create and export annotated scatterplot
    df = df[df['alignment_difference'].notna()].copy()
    df = df[df['word_of_interest_onset'].notna()].copy()
    df = df.sort_values('word_of_interest_onset').reset_index(drop=True)
    df['word_order'] = range(1, len(df) + 1)

    plt.figure(figsize=(10, 6))
    plt.scatter(df['alignment_difference'], df['word_order'], c='blue')

    for _, row in df.iterrows():
        plt.text(
            row['alignment_difference'] + 0.05,
            row['word_order'] + 0.02,
            row['word_of_interest'],
            fontsize=13
        )

    x_range = df['alignment_difference'].max() - df['alignment_difference'].min()
    y_range = df['word_order'].max() - df['word_order'].min()
    x_padding = x_range * 0.06
    y_padding = y_range * 0.1

    plt.xlim(df['alignment_difference'].min() - x_padding, df['alignment_difference'].max() + x_padding)
    plt.ylim(df['word_order'].min() - y_padding, df['word_order'].max() + y_padding)

    plt.xlabel('Time Difference (Apex Onset - Pronoun Onset) in ms', fontsize=14)
    plt.ylabel('Pronoun-Gesture Pair', fontsize=14)
    plt.yticks(df['word_order'], df['word_order'])
    plt.title('Temporal Alignment of Gestures and Pronouns', pad=20, fontsize=16)

    scatterplot_path = os.path.join(output_dir, f"{base_name}_scatterplot.png")
    plt.tight_layout()
    plt.savefig(scatterplot_path)
    plt.close()

    # Step 7: Merge audio and video
    merged_video_path = os.path.join(output_dir, f"{base_name}_merged.mp4")
    merge_audio_video(video_path, audio_path, merged_video_path)

    # Step 8: Add captions to video
    captions = []
    for segment in transcription_results:
        if 'result' in segment and segment['result']:
            start_time = segment['result'][0].get('start', 0)
            end_time = segment['result'][-1].get('end', 0)
            text = segment.get('text', '')
            captions.append((start_time, end_time, text))

    captioned_video_path = os.path.join(output_dir, f"{base_name}_captioned.mp4")
    add_captions(merged_video_path, captions, captioned_video_path)

    # Step 9: Annotate gesture apices on video
    final_video_path = os.path.join(output_dir, f"{base_name}_final.mp4")
    signal_gesture_peaks(captioned_video_path, gesture_apex_times, final_video_path)

    logging.info(f"Finished processing pair: {base_name}")
