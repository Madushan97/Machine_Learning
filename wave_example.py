# Audio file formats
# .mp3
# .flac
# .wav

import wave

obj = wave.open("01-basics_output.wav", "rb")

print("Number of channels :", obj.getnchannels())
print("sample width :", obj.getsampwidth())
print("frame rate :", obj.getframerate())
print("Number of frames :", obj.getnframes())
print("parameters", obj.getparams())

T_audio = obj.getnframes() / obj.getframerate()
print(T_audio)

# to read all the frames
frames = obj.readframes(-1)

print(type(frames), type(frames[0]))
print(len(frames)/ 2)

obj.close()

obj_new = wave.open("01-basics_output.wav", "wb")

obj_new.setnchannels(1)
obj_new.setsampwidth(2)
obj_new.setframerate(16000.0)

obj_new.writeframes(frames)

obj_new.close()
