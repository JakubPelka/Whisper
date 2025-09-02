import os
import subprocess
from pathlib import Path

def convert_audio_to_wav(input_folder):
    # Loop through all files in the input folder
    for file_path in Path(input_folder).glob("*"):
        if file_path.suffix.lower() in [".mp3", ".m4a"]:
            output_file = file_path.with_suffix(".wav")
            try:
                print(f"Converting {file_path} to {output_file}...")
                # Run ffmpeg command
                subprocess.run(
                    ["ffmpeg", "-i", str(file_path), "-ar", "16000", "-ac", "1", str(output_file)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print(f"Successfully converted: {file_path} -> {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {file_path}: {e}")

# Set the folder path containing the audio files
input_folder = r"C:\Users\jakpel\OneDrive - Kungsbacka kommun\Dokument\Ljudinspelningar"

# Convert all .m4a and .mp3 files in the folder to .wav
convert_audio_to_wav(input_folder)
