import os
import speech_recognition as sr
import pyttsx3
import requests

# 🎤 Setup voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    print("🤖:", text)
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎧 Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("🗣️ You:", text)
            return text
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that.")
        except sr.RequestError:
            speak("Speech service is down.")
        return ""

def get_response(question):
    # Send question to the Flask backend
    try:
        response = requests.post('http://localhost:5000/chat', json={'question': question})
        if response.status_code == 200:
            return response.json()['response']
        else:
            return "Sorry, I couldn't get a response from the server."
    except Exception as e:
        return f"Error: {str(e)}"

# 🔁 Voice assistant loop
speak("Hi, I am your finance voice assistant. How can I help you today?")
while True:
    question = listen()
    if question.lower() in ["exit", "quit", "stop"]:
        speak("Goodbye! Have a great day!")
        break
    if question:
        response = get_response(question)
        speak(response) 