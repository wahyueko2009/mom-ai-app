from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import whisper
import tempfile
import logging
import warnings
import librosa
import re
import openai
from file_handlers import (
    allowed_file, get_file_size_mb, process_audio_file,
    save_transcript, save_mom, UPLOAD_FOLDER, MAX_CONTENT_LENGTH
)
from prompt_handlers import (
    format_transcript, generate_mom_from_transcript
)
from api_config import OPENAI_API_KEY

# Filter peringatan FP16
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['MAX_CONTENT_PATH'] = None  # No path length limit

# Pastikan folder uploads ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inisialisasi model Whisper
model = whisper.load_model("base")

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

# Format hasil transkripsi menjadi paragraf
def format_transcript(text):
    # Hapus kata-kata yang tidak perlu seperti "oke", "ya", dll
    text = re.sub(r'\b(oke|ya|ya ya|silahkan)\b', '', text, flags=re.IGNORECASE)
    
    # Bersihkan spasi berlebih
    text = re.sub(r'\s+', ' ', text)
    
    # Pisahkan berdasarkan kalimat
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # Filter kalimat kosong atau terlalu pendek
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    
    # Format ulang dengan indentasi dan pengelompokan
    formatted_text = []
    current_speaker = None
    
    for sentence in sentences:
        # Deteksi pembicara baru (biasanya dimulai dengan "Pak" atau nama)
        if re.match(r'^(Pak|Bu|Ibu|Bapak|Sdr\.)', sentence, re.IGNORECASE):
            if current_speaker:
                formatted_text.append('')  # Tambah baris kosong antar pembicara
            current_speaker = sentence.split()[0]
            formatted_text.append(f"{sentence}")
        else:
            if current_speaker:
                formatted_text.append(f"    {sentence}")
            else:
                formatted_text.append(sentence)
    
    return '\n'.join(formatted_text)

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def generate_mom_from_transcript(transcript_text):
    logger.info(f"Panjang transkripsi awal: {len(transcript_text)} karakter")
    
    # Split text into chunks if it's too long
    chunks = split_text_into_chunks(transcript_text)
    logger.info(f"Jumlah chunks yang dibuat: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1} panjang: {len(chunk)} karakter")
    
    all_mom_parts = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Memproses chunk {i+1} dari {len(chunks)}")
        prompt = (
            "Buatkan notulen rapat (Minutes of Meeting) yang terstruktur dan informatif dari transkrip berikut. "
            "Format notulen harus mencakup:\n"
            "1. Tanggal dan Waktu Rapat\n"
            "2. Peserta Rapat\n"
            "3. Agenda Rapat\n"
            "4. Pembahasan dan Keputusan\n"
            "5. Tindak Lanjut\n"
            "6. Catatan Penting\n\n"
            "Transkrip rapat:\n"
            f"{chunk}\n"
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Kamu adalah asisten profesional yang ahli dalam membuat notulen rapat yang terstruktur, jelas, dan informatif. Kamu harus memastikan semua informasi penting dari rapat tercatat dengan baik dan mudah dipahami."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.3
        )
        mom_part = response.choices[0].message.content.strip()
        logger.info(f"Panjang hasil chunk {i+1}: {len(mom_part)} karakter")
        all_mom_parts.append(mom_part)
    
    # Gabungkan semua bagian MoM
    combined_mom = "\n\n".join(all_mom_parts)
    logger.info(f"Panjang hasil gabungan: {len(combined_mom)} karakter")
    
    # Buat ringkasan final dari semua bagian
    final_prompt = (
        "Buatkan notulen rapat final yang menggabungkan dan meringkas semua informasi berikut. "
        "Pastikan tidak ada informasi yang terlewat dan hasilnya tetap terstruktur:\n\n"
        f"{combined_mom}"
    )
    
    logger.info("Memproses ringkasan final")
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Kamu adalah asisten profesional yang ahli dalam membuat notulen rapat yang terstruktur, jelas, dan informatif. Kamu harus memastikan semua informasi penting dari rapat tercatat dengan baik dan mudah dipahami."},
            {"role": "user", "content": final_prompt}
        ],
        max_tokens=4000,
        temperature=0.3
    )
    
    final_result = final_response.choices[0].message.content.strip()
    logger.info(f"Panjang hasil final: {len(final_result)} karakter")
    
    return final_result

# Meningkatkan batas karakter untuk setiap chunk
MAX_CHARS_OPENAI = 24000  # Meningkatkan batas aman karakter untuk input ke OpenAI

def split_text_into_chunks(text, max_chars=MAX_CHARS_OPENAI):
    # Split berdasarkan paragraf untuk menjaga konteks
    paragraphs = text.split('\n\n')
    logger.info(f"Jumlah paragraf yang ditemukan: {len(paragraphs)}")
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, paragraph in enumerate(paragraphs):
        paragraph_length = len(paragraph)
        if current_length + paragraph_length > max_chars:
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                logger.info(f"Chunk baru dibuat dengan panjang {len(chunk_text)} karakter")
                chunks.append(chunk_text)
            current_chunk = [paragraph]
            current_length = paragraph_length
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length
    
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        logger.info(f"Chunk terakhir dibuat dengan panjang {len(chunk_text)} karakter")
        chunks.append(chunk_text)
    
    return chunks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.debug("Menerima request upload file")
        
        if 'audioFile' not in request.files:
            logger.error("Tidak ada file dalam request")
            return jsonify({
                'status': 'error',
                'error': 'Tidak ada file yang diupload'
            }), 400
        
        file = request.files['audioFile']
        if file.filename == '':
            logger.error("Nama file kosong")
            return jsonify({
                'status': 'error',
                'error': 'Tidak ada file yang dipilih'
            }), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Tipe file tidak diizinkan: {file.filename}")
            return jsonify({
                'status': 'error',
                'error': 'Tipe file tidak diizinkan. Gunakan format MP3, WAV, M4A, atau MP4'
            }), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.debug(f"Menyimpan file ke: {filepath}")
        file.save(filepath)
        
        # Validasi ukuran file setelah disimpan
        file_size_mb = get_file_size_mb(filepath)
        logger.debug(f"Ukuran file: {file_size_mb:.2f}MB")
        
        if file_size_mb > 100:
            logger.error(f"File terlalu besar: {file_size_mb:.2f}MB")
            os.remove(filepath)  # Hapus file jika terlalu besar
            return jsonify({
                'status': 'error',
                'error': f'Ukuran file ({file_size_mb:.2f}MB) melebihi batas maksimum (100MB)'
            }), 400
        
        try:
            # Proses file audio
            result = process_audio_file(filepath)
            transcript = result['transcript']
            transcript_formatted = format_transcript(transcript)
            
            # Simpan hasil transkripsi
            txt_filename = save_transcript(transcript_formatted, filename, app.config['UPLOAD_FOLDER'])
            
            # Generate MoM
            try:
                mom = generate_mom_from_transcript(transcript_formatted)
                mom_filename = save_mom(mom, txt_filename, app.config['UPLOAD_FOLDER'])
            except Exception as e:
                mom = None
                mom_filename = None
                logger.error(f"Gagal generate MoM: {str(e)}")
            
            return jsonify({
                'status': 'success',
                'message': f'File berhasil diproses (Ukuran: {file_size_mb:.2f}MB, Durasi: {result["duration"]} menit)',
                'transcript': transcript,
                'txt_file': txt_filename,
                'file_size': f'{file_size_mb:.2f}MB',
                'duration': f'{result["duration"]} menit',
                'mom': mom,
                'mom_file': mom_filename
            })
            
        except Exception as e:
            logger.error(f"Error saat memproses file: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'status': 'error',
                'error': f'Terjadi kesalahan saat memproses file: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error umum: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Terjadi kesalahan: {str(e)}'
        }), 500

@app.route('/generate_mom', methods=['POST'])
def generate_mom():
    data = request.json
    txt_filename = data.get('txt_file')
    if not txt_filename:
        return jsonify({'status': 'error', 'error': 'Nama file txt tidak diberikan'}), 400

    txt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)
    if not os.path.exists(txt_filepath):
        return jsonify({'status': 'error', 'error': 'File txt tidak ditemukan'}), 404

    with open(txt_filepath, 'r', encoding='utf-8') as f:
        transcript_text = f.read()

    try:
        mom = generate_mom_from_transcript(transcript_text)
        mom_filename = save_mom(mom, txt_filename, app.config['UPLOAD_FOLDER'])
        return jsonify({'status': 'success', 'mom': mom, 'mom_file': mom_filename})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_uploads():
    try:
        # Hapus semua file di folder uploads
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error menghapus file {filename}: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'error': f'Gagal menghapus beberapa file: {str(e)}'
                }), 500
        
        return jsonify({
            'status': 'success',
            'message': 'Folder uploads berhasil dibersihkan'
        })
    except Exception as e:
        logger.error(f"Error saat membersihkan folder: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Terjadi kesalahan: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 