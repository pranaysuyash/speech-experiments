import os
import wave


def generate_large_wav(filename, size_mb=100):
    """Generates a dummy WAV file of approximately size_mb MB."""
    target_size_bytes = size_mb * 1024 * 1024
    sample_rate = 44100
    num_channels = 2
    sampwidth = 2

    # Calculate number of frames needed
    bytes_per_frame = num_channels * sampwidth
    num_frames = target_size_bytes // bytes_per_frame

    print(f"Generating {filename} with size ~{size_mb} MB ({num_frames} frames)...")

    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)

        # Write chunks to avoid memory issues
        chunk_size = 100000
        frames_written = 0

        while frames_written < num_frames:
            current_chunk = min(chunk_size, num_frames - frames_written)
            # Generate random noise
            data = bytearray(os.urandom(current_chunk * bytes_per_frame))
            wav_file.writeframes(data)
            frames_written += current_chunk

    print(f"Done. Created {filename}")


if __name__ == "__main__":
    generate_large_wav("inputs/abuse_large_file.wav", size_mb=100)
