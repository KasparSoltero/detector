import os
import wave

def get_duration(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        duration = frames / float(rate)
        return duration

def print_longer_than_10_seconds(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            duration = get_duration(file_path)
            if duration > 10:
                print(file_name)

# Replace '/path/to/directory' with the actual directory path
directory_path = 'data/manually_isolated/bird'
print_longer_than_10_seconds(directory_path)