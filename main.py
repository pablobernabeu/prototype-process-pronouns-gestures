
import argparse
import os
import webvtt
import glob
import logging
import pandas as pd
import matplotlib.pyplot as plt

from audio_processing import transcribe_audio
from video_processing import detect_multiple_gesture_apexes
from alignment_analysis import extract_word_of_interest_onsets, calculate_alignment
from video_editing import merge_audio_video, add_captions, signal_gesture_peaks

def process_pair(audio_path, video_path, model_path, output_dir, max_time_diff, words_of_interest):
    '''
    Processes a matching audio and video file pair.
    # 1. Transcribe audio
    # 2. Extract word onsets
    # 3. Detect gesture apexes
    # 4. Compute alignment differences
    # 5. Prepare context for each word onset
    # 6. Plot alignment differences
    # 7. Merge audio into video
    # 8. Add captions
    # 9. Signal gesture peaks on video
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
    word_onsets = extract_word_of_interest_onsets(transcription_results, words_of_interest)
    
    # Step 3: Detect one or multiple gesture apexes
    gesture_apex_times = detect_multiple_gesture_apexes(video_path)  
    
    # Step 4: Compute alignment difference
    df_rows = []
    used_apexes = set()  
    
    # Step 5: Prepare correct context extraction

    df_rows = []
    used_apexes = set()

    # Create a structured list of transcript chunks [(start_time, end_time, text)]
    transcript_chunks = [
        (segment['result'][0]['start'], segment['result'][-1]['end'], segment['text'])
        for segment in transcription_results if 'result' in segment and segment['result']
    ]

    for word_of_interest, onset_time in word_onsets:
        matched_context = None

        # Find the segment where the target word appears
        for start_time, end_time, sentence in transcript_chunks:
            if start_time <= onset_time <= end_time:
                words = sentence.split()
                if word_of_interest in words:
                    index = words.index(word_of_interest)
                    prev_word = words[index - 1] if index > 0 else ""
                    next_word = words[index + 1] if index < len(words) - 1 else ""
                    matched_context = f"{prev_word} {word_of_interest} {next_word}".strip()
                    break  # Stop at the first valid match

        if not matched_context:
            matched_context = word_of_interest  # Fallback if no match is found

        # Find the closest gesture apex
        available_apexes = [
            apex for apex in gesture_apex_times
            if apex not in used_apexes and abs(onset_time - apex) <= max_time_diff
        ]

        if available_apexes:
            nearest_apex = min(available_apexes, key=lambda apex: abs(onset_time - apex))
            used_apexes.add(nearest_apex)
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
    
    # Convert to a DataFrame
    df = pd.DataFrame(df_rows)
    
    # Export to CSV
    csv_path = os.path.join(output_dir, f"{base_name}_alignment.csv")
    df.to_csv(csv_path, index=False)
    print(f"Alignment data saved to {csv_path}")
    
    # Step 6: Plot

    # Filter rows with valid alignment differences
    df = df[df['alignment_difference'].notna()].copy()

    # Ensure required column is present
    if 'word_of_interest_onset' not in df.columns:
        raise KeyError("'word_of_interest_onset' column is missing from the dataframe.")

    # Drop rows with missing onset values
    df = df[df['word_of_interest_onset'].notna()].copy()

    # Sort by word onset to assign consistent numeric order
    df = df.sort_values('word_of_interest_onset').reset_index(drop=True)
    df['word_order'] = range(1, len(df) + 1)

    # Check that word_order has been successfully created
    if 'word_order' not in df.columns:
        raise KeyError("Failed to create 'word_order' column.")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['alignment_difference'], df['word_order'], c='blue')

    # Add text labels with reduced padding
    for i, row in df.iterrows():
        plt.text(
            row['alignment_difference'] + 0.05,  # X offset
            row['word_order'] + 0.02,            # Y offset
            row['word_of_interest'],
            fontsize=13
        )

    # Add padding to axis limits
    x_range = df['alignment_difference'].max() - df['alignment_difference'].min()
    y_range = df['word_order'].max() - df['word_order'].min()

    x_padding = x_range * 0.06  # 6% of range on each side
    y_padding = y_range * 0.1   # 10% of range on each side

    plt.xlim(df['alignment_difference'].min() - x_padding,
             df['alignment_difference'].max() + x_padding)

    plt.ylim(df['word_order'].min() - y_padding,
             df['word_order'].max() + y_padding)

    # Axis labels
    plt.xlabel('Time Difference (Apex Onset - Pronoun Onset) in ms', fontsize=14)
    plt.ylabel('Pronoun-Gesture Pair', fontsize=14)
    
    # Y-ticks: map word_order to itself
    plt.yticks(df['word_order'], df['word_order'])

    # Title with padding below
    plt.title('Temporal Alignment of Gestures and Pronouns', pad=20, fontsize=16)

    scatterplot_path = os.path.join(output_dir, f"{base_name}_scatterplot.png")
    plt.tight_layout()
    plt.savefig(scatterplot_path)
    plt.close()
    
    # Step 7: Merge audio into video.
    merged_video_path = os.path.join(output_dir, f"{base_name}_merged.mp4")
    merge_audio_video(video_path, audio_path, merged_video_path)
    
    # Step 8: Add captions
    captions = []
    for segment in transcription_results:
        if 'result' in segment and segment['result']:
            start_time = segment['result'][0].get('start', 0)
            end_time = segment['result'][-1].get('end', 0)
            text = segment.get('text', '')
            captions.append((start_time, end_time, text))
    
    captioned_video_path = os.path.join(output_dir, f"{base_name}_captioned.mp4")
    add_captions(merged_video_path, captions, captioned_video_path)
    
    # Step 9: Signal gesture peaks on the video.
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
    parser.add_argument('--words_of_interest', required=True, help='Comma-separated list of target words (e.g. "this,that,these,those")')
    parser.add_argument('--output', required=True, help='Output directory for processed files')
    parser.add_argument('--max_time_diff', type=int, default=2000, help='Maximum time difference (ms) between target word onset and gesture apex')  
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_arguments()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    audio_files = sorted(glob.glob(os.path.join(args.audio_folder, '*.wav')))
    if not audio_files:
        logging.error("No audio files found in the specified folder.")
        return
    
    for audio_file in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        video_file = os.path.join(args.video_folder, f"{base_name}.mp4")
        if not os.path.exists(video_file):
            logging.warning(f"Matching video file for {base_name} not found. Skipping.")
            continue
        process_pair(audio_file, video_file, args.model, args.output, args.max_time_diff, args.words_of_interest.split(','))
    
if __name__ == '__main__':
    main()
