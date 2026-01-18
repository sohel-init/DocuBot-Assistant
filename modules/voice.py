import speech_recognition as sr
import io
from groq import Groq
from config import Config

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

def transcribe_audio_with_groq(audio_bytes):
    """
    Transcribes audio using Groq's Whisper API.
    """    
    client = Groq(api_key=Config.GROQ_API_KEY)
    
    try:
        # Wrap the raw bytes in a BytesIO object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm" 
        
        # Send to Groq Whisper
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=Config.GROQ_WHISPER_MODEL_NAME,
            response_format="text"
        )
        
        return transcription
        
    except Exception as e:
        return f"Transcription Error: {str(e)}"
