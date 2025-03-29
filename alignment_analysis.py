
def extract_target_word_onsets(results, demonstratives=None):
    """
    Extracts the onset times of demonstrative pronouns from the recogniser results,
    returning a list of (word, onset) tuples.
    """
    word_onsets = []
    for segment in results:
        if 'result' in segment and isinstance(segment['result'], list):
            for word_info in segment['result']:
                if 'word' in word_info and 'start' in word_info:
                    word = word_info['word'].lower()
                    if word in demonstratives:
                        onset_time = word_info['start'] * 1000 # Convert to ms
                        # Return a tuple of (word, onset_time)
                        word_onsets.append((word, onset_time))
    return word_onsets

def calculate_alignment(demonstrative_onsets, gesture_apex_times):
    '''
    Calculates the temporal differences between each demonstrative onset and the closest gesture apex.
    Returns a list of raw differences in milliseconds (not absolute values).
    '''
    # Convert gesture_apex_times to ms
    gesture_apex_times = [apex * 1000 for apex in gesture_apex_times]  # Convert to ms
    
    alignments = []
    for onset in demonstrative_onsets:
        differences = [onset - apex for apex in gesture_apex_times]
        if differences:
            alignments.append(min(differences))
        else:
            alignments.append(None)
    return alignments

