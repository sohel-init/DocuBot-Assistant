import speech_recognition as sr
import io

def transcribe_audio(audio_bytes):
    """
    Transcribes audio bytes to text using Google Speech Recognition
    """
    recognizer = sr.Recognizer()
    
    # Convert bytes to a file-like object that AudioFile can read
    audio_file = io.BytesIO(audio_bytes)
    
    try:
        # Load the audio file into speech_recognition
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            
        # Transcribe
        text = recognizer.recognize_google(audio_data)
        return text
        
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return f"API Error: {e}"
    except Exception as e:
        return f"Error: {e}"
