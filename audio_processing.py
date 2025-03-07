
import json
import wave
import logging
from vosk import Model, KaldiRecognizer

def transcribe_audio(audio_path, model_path):
    '''
    Transcribes the audio file using Vosk and returns a list of recognition results.
    '''
    try:
        logging.info('Initialising Vosk model…')
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
        
        final_result = json.loads(recogniser.FinalResult())
        results.append(final_result)
        logging.info('Transcription completed.')
        return results
    except Exception as e:
        logging.error(f'Error in transcribing audio: {e}')
        return []
