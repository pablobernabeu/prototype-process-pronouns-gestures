from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings
import conf  # Import the conf.py file
change_settings({"IMAGEMAGICK_BINARY": conf.IMAGEMAGICK_BINARY})
import logging

def merge_audio_video(video_path, audio_path, output_path):
    '''
    Merges audio into the video using MoviePy.
    '''
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        final_video = video.set_audio(audio)
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        logging.info('Audio and video merged successfully.')
    except Exception as e:
        logging.error(f'Error in merging audio and video: {e}')

def add_captions(video_path, captions, output_path):
    '''
    Adds captions to the video.
    Captions is a list of tuples (start_time_ms, end_time_ms, text).
    '''
    try:
        video = VideoFileClip(video_path)
        caption_clips = []
        for start_ms, end_ms, text in captions:
            start_sec = start_ms / 1000  # ✅ Convert ms to seconds
            end_sec = end_ms / 1000  # ✅ Convert ms to seconds
            duration_sec = end_sec - start_sec  # ✅ Calculate duration in seconds

            txt_clip = TextClip(text, fontsize=24, color='white', bg_color='black')
            txt_clip = txt_clip.set_position(('center', 'bottom')).set_start(start_sec).set_duration(duration_sec)
            caption_clips.append(txt_clip)

        final = CompositeVideoClip([video] + caption_clips)
        final.write_videofile(output_path, codec='libx264', audio_codec='aac')
        logging.info('Captions added to video successfully.')
    except Exception as e:
        logging.error(f'Error in adding captions: {e}')

def signal_gesture_peaks(video_path, gesture_peaks_ms, output_path):
    '''
    Signals gesture peaks in the video by overlaying markers on the frames corresponding to gesture peaks.
    gesture_peaks_ms is a list of times (in milliseconds) indicating gesture apex.
    '''
    try:
        video = VideoFileClip(video_path)
        marker_clips = []
        for peak_ms in gesture_peaks_ms:
            peak_sec = peak_ms / 1000  # ✅ Convert ms to seconds

            marker = TextClip('GESTURE PEAK', fontsize=30, color='red', bg_color='yellow')
            marker = marker.set_position(('center', 'top')).set_start(peak_sec).set_duration(0.5)
            marker_clips.append(marker)

        final = CompositeVideoClip([video] + marker_clips)
        final.write_videofile(output_path, codec='libx264', audio_codec='aac')
        logging.info('Gesture peaks signalled successfully.')
    except Exception as e:
        logging.error(f'Error in signalling gesture peaks: {e}')
