import sounddevice as sd
import numpy as np
import soundfile as sf
from vosk import Model, KaldiRecognizer

def record_audio(duration=5, sample_rate=16000):
    """Record audio from the microphone."""
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until the recording is finished
    print("Recording complete.")
    return audio_data.flatten()

def save_audio_to_file(audio_data, sample_rate, filename='output.wav'):
    """Save audio data to a WAV file."""
    sf.write(filename, audio_data, sample_rate)

def transcribe_audio(filename, model_path):
    """Transcribe audio using Vosk."""
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    with sf.SoundFile(filename) as audio_file:
        while True:
            data = audio_file.read(4000, dtype=np.int16)
            if len(data) == 0:
                break
            # Convert numpy array to bytes
            data_bytes = data.tobytes()
            if recognizer.AcceptWaveform(data_bytes):
                result = recognizer.Result()
                print(result)

        final_result = recognizer.FinalResult()
        print(final_result)

if __name__ == "__main__":
    duration = 5  # seconds
    sample_rate = 16000  # Hz
    model_path = "vosk-model-small-en-us-0.15"  # Path to your Vosk model directory

    # Record audio
    audio_data = record_audio(duration, sample_rate)
    
    # Save the recorded audio to a file
    save_audio_to_file(audio_data, sample_rate)
    
    # Transcribe the saved audio file
    transcribe_audio('output.wav', model_path)

