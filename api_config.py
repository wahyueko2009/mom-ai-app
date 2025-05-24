
import openai
import os
from dotenv import load_dotenv

# Muat variabel dari .env
load_dotenv()

# Ambil API key dari environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Konfigurasi OpenAI
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Batas karakter input ke OpenAI
MAX_CHARS_OPENAI = 12000

