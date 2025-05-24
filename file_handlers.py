import os
from werkzeug.utils import secure_filename
import whisper
import librosa
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inisialisasi model Whisper
model = whisper.load_model("base")

# Konfigurasi upload
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'mp4'}
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def process_audio_file(filepath):
    """Process audio file and return transcript"""
    try:
        # Transcribe audio menggunakan Whisper
        result = model.transcribe(filepath, language='id')
        transcript = result["text"]
        
        # Hitung durasi audio (dalam detik)
        duration = librosa.get_duration(path=filepath)
        duration_minutes = round(duration / 60, 1)
        
        return {
            'transcript': transcript,
            'duration': duration_minutes
        }
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise

def save_transcript(transcript, filename, upload_folder):
    """Save transcript to file"""
    txt_filename = os.path.splitext(filename)[0] + '.txt'
    txt_filepath = os.path.join(upload_folder, txt_filename)
    
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    return txt_filename

def save_mom(mom_text, txt_filename, upload_folder):
    """Save MoM to file"""
    mom_filename = txt_filename.replace('.txt', '.mom.txt')
    mom_filepath = os.path.join(upload_folder, mom_filename)
    
    with open(mom_filepath, 'w', encoding='utf-8') as f:
        f.write(mom_text)
    
    return mom_filename 