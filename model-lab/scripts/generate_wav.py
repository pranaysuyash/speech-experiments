import wave

# Generate 1 second of silence
sample_rate = 16000
duration = 1.0
num_samples = int(sample_rate * duration)

with wave.open("inputs/valid_test.wav", "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    f.writeframes(b"\x00\x00" * num_samples)

print("Created inputs/valid_test.wav")
