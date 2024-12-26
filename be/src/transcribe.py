import whisper
import os
from pydub import AudioSegment
from pydub.effects import normalize

# Load the Whisper model (use a larger model if possible)
model = whisper.load_model("large")

# Normalize and preprocess the audio
def preprocess_audio(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    normalized_audio = normalize(audio)
    normalized_audio.export(output_file, format="wav")

# Transcribe the audio
def transcribe_audio(file_path, output_file, language="ne"):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} does not exist.")
        return

    # Transcribe the audio
    result = model.transcribe(file_path, language=language)

    # Get the transcription text
    transcription = result['text']

    # Print and save the transcription
    print(f"Transcription: {transcription}")
    with open(output_file, 'w') as f:
        f.write(transcription)
    print(f"Transcription saved to {output_file}")

# Example usage
audio_file = "./baby.m4a"  # Replace with your audio file's path
normalized_audio = "./baby.wav"
output_file = "./baby.txt"  # The path to save the transcription

# Preprocess the audio (normalization)
preprocess_audio(audio_file, normalized_audio)

# Transcribe the preprocessed audio
transcribe_audio(normalized_audio, output_file)
