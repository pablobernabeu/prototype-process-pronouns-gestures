import argparse
import os
import webvtt
import glob
import logging
import pandas as pd
import matplotlib.pyplot as plt

from audio_processing import transcribe_audio
from video_processing import detect_multiple_gesture_apexes
from alignment_analysis import extract_demonstrative_onsets, calculate_alignment
from video_editing import merge_audio_video, add_captions, signal_gesture_peaks

def process_pair(audio_path, video_path, model_path, output_dir):
    '''
    Processes a matching audio and video file pair.
    1) Transcribes audio and extracts demonstrative onsets.
    2) Detects gesture apex(es).
    3) Calculates alignment.
    4) Exports CSV alignment data.
    5) Merges audio/video, adds captions, signals gesture peaks.
    6) Now also exports a plain text transcription and a WebVTT file.
    '''
    
    # Step 1: Transcribe audio
    transcription_results = transcribe_audio(audio_path, model_path)
    
    # Export transcription to a txt file
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}_transcription.txt")
    
    # Gather all 'text' fields from the transcription results
    all_text_segments = []
    for segment in transcription_results:
        seg_text = segment.get('text', '')
        if seg_text:
            all_text_segments.append(seg_text)
    
    # Join them with a newline or space
    full_transcription = "\n".join(all_text_segments)
    
    # Write to .txt
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_transcription)
    
    print(f"Plain text transcription saved to {txt_path}")
    
    # Export transcription to a WebVTT subtitle file.
    
    # Helper function to convert float seconds to WebVTT time format.
    def sec_to_timestamp(seconds):
        import math
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        msecs = int(round((secs - int(secs)) * 1000))
        secs_int = int(secs)
        # Format: HH:MM:SS.mmm
        return f"{hours:02d}:{minutes:02d}:{secs_int:02d}.{msecs:03d}"
    
    vtt = webvtt.WebVTT()
    
    # Each segment with 'result' typically has a list of words with start/end times
    for segment in transcription_results:
        # We'll create one caption per segment if it has timestamps
        if 'result' in segment and segment['result']:
            start_time = segment['result'][0].get('start', 0)
            end_time = segment['result'][-1].get('end', start_time)
            text = segment.get('text', '')
            
            # Only add a caption if there's actual text
            if text:
                caption = webvtt.Caption(
                    start=sec_to_timestamp(start_time),
                    end=sec_to_timestamp(end_time),
                    text=text
                )
                vtt.captions.append(caption)
    
    vtt_path = os.path.join(output_dir, f"{base_name}.vtt")
    vtt.save(vtt_path)
    print(f"WebVTT captions saved to {vtt_path}")
    
    # Step 2: Extract (word, onset) pairs
    word_onsets = extract_demonstrative_onsets(transcription_results)
    
    # Step 3: Detect one or multiple gesture apexes
    gesture_apex_times = detect_multiple_gesture_apexes(video_path)  
    
    # Step 4: For each demonstrative onset, find the nearest apex using absolute difference 
    # and compute alignment difference using raw difference.
    df_rows = []
    for (demonstrative_word, onset_time) in word_onsets:
        if gesture_apex_times:
            nearest_apex = min(gesture_apex_times, key=lambda apex: abs(onset_time - apex))
            alignment_diff = onset_time - nearest_apex
        else:
            nearest_apex = None
            alignment_diff = None
        
        df_rows.append({
            'demonstrative_word': demonstrative_word,
            'demonstrative_onset': onset_time,
            'gesture_apex': nearest_apex,
            'alignment_difference': alignment_diff
        })
    
    # Convert to a DataFrame
    df = pd.DataFrame(df_rows)
    
    # Export to CSV
    csv_path = os.path.join(output_dir, f"{base_name}_alignment.csv")
    df.to_csv(csv_path, index=False)
    print(f"Alignment data saved to {csv_path}")
    
    # Plot and save histogram.
    plt.figure(figsize=(10, 6))
    plt.hist(df['alignment_difference'], bins=10, edgecolor='black')
    plt.title('Distribution of Temporal Differences')
    plt.xlabel('Temporal Difference (ms)')
    plt.ylabel('Frequency')
    hist_path = os.path.join(output_dir, f"{base_name}_hist.png")
    plt.savefig(hist_path)
    plt.close()
    
    # Plot and save scatter plot.
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['demonstrative_onset'], df['alignment_difference'], c='blue')
        
    # Annotate each point with the demonstrative_word
    for i, row in df.iterrows():
        plt.text(
            row['demonstrative_onset'] + 1.5,  # small offset to the right
            row['alignment_difference'] + 1.5, # small offset above
            row['demonstrative_word'],
            fontsize=8
        )

    plt.title('Demonstrative Onset vs Temporal Difference')
    plt.xlabel('Demonstrative Onset (ms)')
    plt.ylabel('Temporal Difference (ms)')
    scatter_path = os.path.join(output_dir, f"{base_name}_scatter.png")
    plt.savefig(scatter_path)
    plt.close()
    
    # Step 4: Merge audio into video.
    merged_video_path = os.path.join(output_dir, f"{base_name}_merged.mp4")
    merge_audio_video(video_path, audio_path, merged_video_path)
    
    # Step 5: Add captions from transcription results.
    captions = []
    for segment in transcription_results:
        if 'result' in segment and segment['result']:
            start_time = segment['result'][0].get('start', 0)
            end_time = segment['result'][-1].get('end', 0)
            text = segment.get('text', '')
            captions.append((start_time, end_time, text))
    captioned_video_path = os.path.join(output_dir, f"{base_name}_captioned.mp4")
    add_captions(merged_video_path, captions, captioned_video_path)
    
    # Step 6: Signal gesture peaks on the video.
    final_video_path = os.path.join(output_dir, f"{base_name}_final.mp4")
    signal_gesture_peaks(captioned_video_path, gesture_apex_times, final_video_path)
    
    logging.info(f"Finished processing pair: {base_name}")

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyse speech and gesture alignment for multiple audio–video pairs.')
    parser.add_argument('--audio_folder', required=True, help='Path to the folder containing audio files (WAV format)')
    parser.add_argument('--video_folder', required=True, help='Path to the folder containing video files (e.g. MP4)')
    parser.add_argument('--model', required=True, help='Path to the Vosk model directory')
    parser.add_argument('--output', required=True, help='Output directory for processed files')
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_arguments()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Find all audio files (e.g. 1.wav, 2.wav, …) in the audio folder.
    audio_files = sorted(glob.glob(os.path.join(args.audio_folder, '*.wav')))
    if not audio_files:
        logging.error("No audio files found in the specified folder.")
        return
    
    for audio_file in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        # Look for a matching video file (e.g. 1.mp4) in the video folder.
        video_file = os.path.join(args.video_folder, f"{base_name}.mp4")
        if not os.path.exists(video_file):
            logging.warning(f"Matching video file for {base_name} not found in {args.video_folder}. Skipping.")
            continue
        process_pair(audio_file, video_file, args.model, args.output)
    
if __name__ == '__main__':
    main()
    