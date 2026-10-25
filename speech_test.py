import speech_recognition as sr

r = sr.Recognizer()

# Replace None with your specific device index
with sr.Microphone(device_index=2) as source:
    print("Adjusting for ambient noise... Please wait.")
    r.adjust_for_ambient_noise(source, duration=1) 
    print("Listening...")
    
    try:
        audio = r.listen(source, timeout=5, phrase_time_limit=5)
        print("Processing speech...")
        text = r.recognize_google(audio)
        print(f"You said: {text}")
    except sr.WaitTimeoutError:
        print("Listening timed out. No speech detected.")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
