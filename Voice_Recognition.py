import speech_recognition
import pyttsx3

recognizer = speech_recognition.Recognizer()

while True:

    try:

        #microphone as an input
        with speech_recognition.Microphone() as mic:

            # start talking, stop talking time = 0.2s
            recognizer.adjust_for_ambient_noise(mic, duration = 0.2)
            audio = recognizer.listen(mic)

            #pick a language etc
            text = recognizer.recognize_google(audio)
            text = text.lower()

            print(f"Recognize {text}")
    
    except speech_recognition.UnknownValueError():

       recognizer = speech_recognition.Recognizer() 
       continue
       