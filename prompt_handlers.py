import re
from api_config import client, MAX_CHARS_OPENAI

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
    current_paragraph = []
    
    for sentence in sentences:
        # Deteksi pembicara baru (biasanya dimulai dengan "Pak" atau nama)
        if re.match(r'^(Pak|Bu|Ibu|Bapak|Sdr\.)', sentence, re.IGNORECASE):
            # Jika ada paragraf sebelumnya, tambahkan ke formatted_text
            if current_paragraph:
                formatted_text.append(' '.join(current_paragraph))
                current_paragraph = []
            
            if current_speaker:
                formatted_text.append('')  # Tambah baris kosong antar pembicara
            
            current_speaker = sentence.split()[0]
            current_paragraph.append(sentence)
        else:
            # Jika kalimat terlalu panjang, buat paragraf baru
            if len(' '.join(current_paragraph + [sentence])) > 200:
                if current_paragraph:
                    formatted_text.append(' '.join(current_paragraph))
                    current_paragraph = []
            current_paragraph.append(sentence)
    
    # Tambahkan paragraf terakhir jika ada
    if current_paragraph:
        formatted_text.append(' '.join(current_paragraph))
    
    # Gabungkan semua paragraf dengan baris kosong di antaranya
    return '\n\n'.join(formatted_text)

def generate_mom_from_transcript(transcript_text):
    prompt = (
        "Buatkan notulen rapat (Minutes of Meeting) yang detail dan terstruktur dari transkrip berikut. "
        "Ikuti panduan berikut:\n\n"
        "1. Analisis konteks rapat dan tentukan format yang paling sesuai\n"
        "2. Identifikasi semua informasi penting termasuk:\n"
        "   - Tanggal, waktu, dan lokasi rapat\n"
        "   - Daftar peserta (jika disebutkan)\n"
        "   - Topik-topik yang dibahas\n"
        "   - Keputusan dan kesepakatan yang diambil\n"
        "   - Tindak lanjut dan deadline (jika ada)\n"
        "   - Catatan khusus atau masalah yang perlu diperhatikan\n"
        "3. Gunakan format yang jelas dan mudah dibaca\n"
        "4. Pastikan semua informasi penting tercatat dengan lengkap\n"
        "5. Jika ada angka, data, atau informasi spesifik, pastikan tercatat dengan akurat\n\n"
        "Transkrip rapat:\n"
        f"{transcript_text}\n"
    )
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "Kamu adalah asisten profesional yang ahli dalam membuat notulen rapat. Kamu harus menganalisis isi rapat secara mendalam, mengidentifikasi semua informasi penting, dan menyajikannya dalam format yang terstruktur dan mudah dipahami. Pastikan tidak ada informasi penting yang terlewat dan semua detail tercatat dengan akurat."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.3
    )
    return response.choices[0].message.content.strip() 