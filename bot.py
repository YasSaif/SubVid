"""SubVid Ai - Telegram bot: burn subtitles into a video or extract them as text.

Flow: pick UI language -> send video/link -> source language (auto/manual)
      -> target language -> output type -> (style settings) -> process.
Pipeline: download -> audio (ffmpeg) -> transcribe (faster-whisper)
          -> translate (Google API / gtx / MyMemory) -> clean + wrap -> burn or send text.
"""
import asyncio
import base64
import gc
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict, deque
from pathlib import Path

import httpx
import yt_dlp
from faster_whisper import WhisperModel
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ----------------------------------------------------------------- config
TOKEN = os.environ["TOKEN"]
PORT = int(os.environ.get("PORT", "10000"))
WEBHOOK_URL = (
    os.environ.get("WEBHOOK_URL")
    or os.environ.get("WEBHOOKURL")
    or os.environ.get("RENDER_EXTERNAL_URL")  # set automatically by Render
    or ""
).rstrip("/")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
MAX_RES = int(os.environ.get("MAX_RES", "480"))  # max size of the SMALLER video side (480 = 480x854 portrait)
USE_VAD = os.environ.get("USE_VAD", "1") != "0"  # "0" skips the VAD model (saves RAM, may add noise text)
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "2"))
MAX_MINUTES = int(os.environ.get("MAX_MINUTES", "10"))          # burn-in video (slow on the free CPU)
MAX_MINUTES_TXT = int(os.environ.get("MAX_MINUTES_TXT", "25"))  # text + SRT only (Groq accepts files up to 25MB)
COOKIES_FILE = os.environ.get("YTDLP_COOKIES")  # optional cookies.txt for yt-dlp
BURN_PRESET = os.environ.get("BURN_PRESET", "ultrafast")  # ultrafast = fastest encode (slightly bigger file)
SS = max(1, min(3, int(os.environ.get("SUB_SUPERSAMPLE", "1"))))  # render subtitles at SSx size, then shrink (avoids glyph boxes at tiny sizes)
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0") or 0)   # your Telegram numeric id (send /id to the bot); receives error alerts
START_TIME = time.time()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # optional: far more accurate transcription + translation
GROQ_ASR_MODEL = os.environ.get("GROQ_ASR_MODEL", "whisper-large-v3")  # most accurate; turbo is used as fallback
GROQ_ASR_FALLBACK = "whisper-large-v3-turbo"
SUB_DELAY = float(os.environ.get("SUB_DELAY", "0.12"))  # seconds subtitles are shifted later (text was appearing before the voice)
GROQ_PROMPT = os.environ.get("GROQ_PROMPT", "")  # optional: names/terms that appear in your clips
GROQ_LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b")  # llama-3.3-70b gave model_not_found on this account
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # optional official Google Cloud Translation key
TG_UPLOAD_LIMIT = 49 * 1024 * 1024  # Bot API upload limit is 50 MB

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("subvid")

URL_RE = re.compile(r"https?://\S+")
JOB_SEM = asyncio.Semaphore(1)  # one heavy job at a time (protects RAM/CPU)
JOBS: "OrderedDict[str, dict]" = OrderedDict()  # key -> job settings
PENDING: list = []  # ids of jobs that are running or waiting (for queue position)


class UserError(Exception):
    """An error whose (localized) message is safe to show to the user."""

    def __init__(self, key: str, **kw):
        super().__init__(key)
        self.key = key
        self.kw = kw


# -------------------------------------------------------------- languages
LANGS = {
    "ar": "العربية", "en": "English", "fr": "Français", "es": "Español",
    "de": "Deutsch", "tr": "Türkçe", "fa": "فارسی", "ur": "اردو",
    "hi": "हिन्दी", "id": "Bahasa Indonesia", "ru": "Русский", "it": "Italiano",
    "pt": "Português", "nl": "Nederlands", "pl": "Polski", "uk": "Українська",
    "he": "עברית", "el": "Ελληνικά", "th": "ไทย", "vi": "Tiếng Việt",
    "ms": "Bahasa Melayu", "bn": "বাংলা", "sv": "Svenska", "ro": "Română",
    "cs": "Čeština", "hu": "Magyar", "fi": "Suomi", "sw": "Kiswahili",
    "ta": "தமிழ்", "zh": "中文", "ja": "日本語", "ko": "한국어",
}
CJK = {"zh", "ja", "ko"}  # no CJK font installed -> source only, not burn targets
TARGETS = [c for c in LANGS if c not in CJK]
SOURCES = list(LANGS)
ARABIC_SCRIPT = {"ar", "fa", "ur"}
RTL_LANGS = {"ar", "fa", "ur", "he"}
PAGE_SIZE = 10


def ordered(codes, first):
    return [first] + [c for c in codes if c != first] if first in codes else list(codes)


# ----------------------------------------------------------------- styles
# /fonttest on the server showed: Noto Sans Arabic -> boxes where "لا" appears, Noto Kufi Arabic -> all boxes.
# Only fonts verified to render correctly are offered (set ENABLE_BROKEN_FONTS=1 to offer all).
# Naskh + Amiri: verified on the server. DejaVu Sans + FreeSerif: verified in a libass test (Arabic incl. "لا").
FONTS_AR = ["Noto Naskh Arabic", "Amiri", "DejaVu Sans", "FreeSerif"]
CANDIDATES_AR_ALL = FONTS_AR + ["Noto Sans Arabic", "Noto Kufi Arabic"]
if os.environ.get("ENABLE_BROKEN_FONTS") == "1":
    FONTS_AR = list(CANDIDATES_AR_ALL)
# Same pixel size looks different per font (measured on the /fonts preview): evened out relative to Noto Naskh.
FONT_SCALE_AR = {"Amiri": 1.45, "DejaVu Sans": 0.62, "FreeSerif": 0.78}
FONT_SCALE_LAT = {"DejaVu Sans": 0.92, "DejaVu Serif": 0.92}


def font_scale(font: str, group: str) -> float:
    return (FONT_SCALE_AR if group == "ar" else FONT_SCALE_LAT).get(font, 1.0)
FONTS_LAT = ["Noto Sans", "Noto Serif", "DejaVu Sans", "DejaVu Serif"]
CANDIDATES_AR = list(CANDIDATES_AR_ALL)   # everything we would like to offer
CANDIDATES_LAT = list(FONTS_LAT)
SIZES = [("size_s", 0.85), ("size_m", 1.2), ("size_l", 1.65)]
STYLES = ["classic", "box", "yellow", "yellowbox", "whitebox", "bluebox", "redbox",
          "green", "cyan", "orange", "pink", "thick", "shadow"]


def _plain(color, outline=2, shadow=0, out_color="&H00000000"):
    return f"PrimaryColour={color},OutlineColour={out_color},BorderStyle=1,Outline={outline},Shadow={shadow}"


def _boxed(text_color, box_color, pad=3):
    return f"PrimaryColour={text_color},OutlineColour={box_color},BackColour={box_color},BorderStyle=3,Outline={pad},Shadow=0"


STYLE_DEFS = {  # ASS colours are &HAABBGGRR (AA=00 opaque, FF transparent)
    "classic": _plain("&H00FFFFFF"),
    "box": "PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BackColour=&H80000000,BorderStyle=3,Outline=3,Shadow=0",
    "yellow": _plain("&H0000FFFF"),
    "yellowbox": _boxed("&H00000000", "&H0000E6FF"),
    "whitebox": _boxed("&H00000000", "&H00FFFFFF"),
    "bluebox": _boxed("&H00FFFFFF", "&H00C8501E"),
    "redbox": _boxed("&H00FFFFFF", "&H001E1EC8"),
    "green": _plain("&H0000FF00"),
    "cyan": _plain("&H00FFFF00"),
    "orange": _plain("&H0000A5FF"),
    "pink": _plain("&H00B469FF"),
    "thick": _plain("&H00FFFFFF", outline=4),
    "shadow": _plain("&H00FFFFFF", outline=1, shadow=3, out_color="&H40000000"),
}
POSITIONS = [("pos_bottom", 2), ("pos_middle", 5), ("pos_top", 8)]


def fonts_for(tgt: str):
    return FONTS_AR if tgt in ARABIC_SCRIPT else FONTS_LAT


# ---------------------------------------------------------------- strings
STR = {
    "ar": {
        "pick_ui": "🌐 اختر لغة الواجهة\nChoose the interface language",
        "welcome": "أهلاً بك في SubVid Ai 🎬\n\nأرسل رابط فيديو (يوتيوب، تيك توك، رابط مباشر) أو ملف فيديو.\nلتغيير لغة الواجهة:\n/language",
        "send_media": "أرسل رابط فيديو أو ملف فيديو.",
        "src_q": "🔍 كيف تبي نحدد لغة الكلام في المقطع؟",
        "src_note": "ℹ️ الاكتشاف التلقائي قد ياخذ وقتاً أطول بقليل، واختيار اللغة يدوياً غالباً أسرع وأدق.",
        "src_auto": "🔍 اكتشاف تلقائي",
        "src_manual": "🌐 تحديد اللغة يدوياً",
        "pick_src": "اختر لغة الكلام في المقطع:",
        "tgt_q": "اختر لغة الترجمة:",
        "mode_q": "اختر نوع الإخراج:",
        "mode_burn": "🎬 دمج الترجمة بالفيديو",
        "mode_txt": "📄 نص الترجمة + ملف SRT",
        "settings": "🎨 تخصيص الترجمة\nاضغط على أي زر لتغيير الخيار، ثم اضغط «ابدأ».",
        "font": "الخط", "size": "الحجم", "style": "النمط", "pos": "المكان",
        "font_preview": "🔤 معاينة الخطوط", "font_preview_cap": "🔤 هذي الخطوط بالترتيب اللي تتغير فيه عند الضغط على زر «الخط»",
        "size_s": "صغير", "size_m": "متوسط", "size_l": "كبير",
        "style_classic": "كلاسيكي", "style_box": "خلفية سوداء", "style_yellow": "أصفر",
        "style_yellowbox": "خلفية صفراء", "style_whitebox": "خلفية بيضاء", "style_bluebox": "خلفية زرقاء", "style_redbox": "خلفية حمراء",
        "style_green": "أخضر", "style_cyan": "سماوي", "style_orange": "برتقالي", "style_pink": "وردي", "style_thick": "إطار سميك", "style_shadow": "ظل ناعم",
        "style_preview": "🎭 معاينة الأنماط", "style_preview_cap": "🎭 الأنماط بالترتيب اللي تتغير فيه عند الضغط على زر «النمط»",
        "pos_bottom": "أسفل", "pos_middle": "وسط", "pos_top": "أعلى",
        "start": "▶️ ابدأ", "cancel": "إلغاء", "cancelled": "تم الإلغاء.",
        "expired": "انتهت صلاحية الطلب، أعد إرسال الفيديو.",
        "queued": "⏳ جاري التجهيز...",
        "queue": "⏳ ترتيبك في الطابور: {n} قبلك.",
        "proc_title": "⏳ جاري المعالجة...",
        "bi": "ثنائية اللغة", "on": "مفعّلة", "off": "متوقفة",
        "quick": "⚡ تشغيل سريع (تلقائي ➜ {tgt}) بإعداداتي السابقة",
        "gl_help": "📚 قاموسك الخاص (يصحّح الكلمات التي يخطئ فيها البوت)\n\nللإضافة اكتب كل كلمة في سطر:\n/glossary الكلمة الخطأ=الكلمة الصحيحة\n/glossary اسم_أو_مصطلح\n\nلحذف كلمة: /glossary del الكلمة\nلمسح الكل: /glossary clear",
        "gl_list": "كلماتك المحفوظة", "gl_none": "القاموس فارغ.", "gl_added": "✅ تمت الإضافة", "gl_deleted": "🗑 تم الحذف", "gl_cleared": "🗑 تم مسح القاموس",
        "card_title": "تم بنجاح", "card_from": "من", "card_to": "إلى", "card_len": "مدة المقطع",
        "card_time": "وقت المعالجة", "card_count": "عدد الترجمات", "card_font": "الخط", "card_words": "الكلمات",
        "step_dl": "تحميل الفيديو", "step_audio": "استخراج الصوت", "step_tr": "تفريغ الكلام",
        "step_trans": "الترجمة", "step_burn": "دمج الترجمة بالفيديو", "step_up": "رفع الفيديو", "step_send": "إرسال النتيجة",
        "secs": "ث",
        "p_dl": "⬇️ جاري تحميل الفيديو...",
        "p_audio": "🎧 جاري استخراج الصوت...",
        "p_tr_auto": "📝 جاري تفريغ الكلام (مع اكتشاف اللغة)...",
        "p_tr": "📝 جاري تفريغ الكلام...",
        "p_trans": "🌐 جاري الترجمة...",
        "p_burn": "🎬 جاري دمج الترجمة بالفيديو...",
        "p_up": "📤 جاري رفع الفيديو...",
        "caption_burn": "✅ تم دمج الترجمة.",
        "caption_txt": "📄 الترجمة",
        "caption_srt": "ملف الترجمة المزمّنة (SRT)",
        "too_big": "⚠️ الفيديو الناتج أكبر من 50MB (حد تيليجرام للبوتات)، هذا ملف الترجمة بدلاً منه.",
        "no_speech": "ما قدرت أرصد كلاماً واضحاً في المقطع.",
        "too_long": "الحد الأقصى لمدة المقطع في هذا الوضع {m} دقيقة. جرّب وضع «نص + SRT» للمقاطع الأطول.",
        "dl_fail": "تعذر تحميل الفيديو من هذا الرابط (قد يكون خاصاً أو محظوراً من السيرفر).",
        "dl_empty": "لم يتم العثور على فيديو في هذا الرابط.",
        "file_big": "تيليجرام يمنع البوت من تنزيل ملفات أكبر من 20MB. ارفع الفيديو على أي موقع وأرسل لي الرابط بدلاً من الملف.",
        "translate_fail": "تعذرت الترجمة حالياً (خدمة الترجمة لا تستجيب). حاول بعد قليل.",
        "timeout": "استغرقت المعالجة وقتاً أطول من المسموح، جرّب مقطعاً أقصر.",
        "generic": "حصل خطأ غير متوقع أثناء المعالجة. حاول مرة ثانية.",
    },
    "en": {
        "pick_ui": "🌐 اختر لغة الواجهة\nChoose the interface language",
        "welcome": "Welcome to SubVid Ai 🎬\n\nSend a video link (YouTube, TikTok, direct link) or a video file.\nChange interface language: /language",
        "send_media": "Send a video link or a video file.",
        "src_q": "🔍 How should we find the spoken language in the clip?",
        "src_note": "ℹ️ Auto-detect may take a little longer; choosing the language manually is usually faster and more accurate.",
        "src_auto": "🔍 Auto-detect",
        "src_manual": "🌐 Choose language",
        "pick_src": "Choose the spoken language:",
        "tgt_q": "Choose the translation language:",
        "mode_q": "Choose the output type:",
        "mode_burn": "🎬 Burn subtitles into video",
        "mode_txt": "📄 Translated text + SRT file",
        "settings": "🎨 Subtitle style\nTap a button to change an option, then press Start.",
        "font": "Font", "size": "Size", "style": "Style", "pos": "Position",
        "font_preview": "🔤 Preview fonts", "font_preview_cap": "🔤 Fonts in the order the «Font» button cycles through them",
        "size_s": "Small", "size_m": "Medium", "size_l": "Large",
        "style_classic": "Classic", "style_box": "Black box", "style_yellow": "Yellow",
        "style_yellowbox": "Yellow box", "style_whitebox": "White box", "style_bluebox": "Blue box", "style_redbox": "Red box",
        "style_green": "Green", "style_cyan": "Cyan", "style_orange": "Orange", "style_pink": "Pink", "style_thick": "Thick outline", "style_shadow": "Soft shadow",
        "style_preview": "🎭 Preview styles", "style_preview_cap": "🎭 Styles in the order the «Style» button cycles through them",
        "pos_bottom": "Bottom", "pos_middle": "Middle", "pos_top": "Top",
        "start": "▶️ Start", "cancel": "Cancel", "cancelled": "Cancelled.",
        "expired": "This request expired, please send the video again.",
        "queued": "⏳ Preparing...",
        "queue": "⏳ Your place in the queue: {n} ahead of you.",
        "proc_title": "⏳ Processing...",
        "bi": "Bilingual", "on": "On", "off": "Off",
        "quick": "⚡ Quick run (auto ➜ {tgt}) with my last settings",
        "gl_help": "📚 Your glossary (fixes words the bot gets wrong)\n\nAdd one entry per line:\n/glossary wrong word=correct word\n/glossary Name_or_term\n\nDelete: /glossary del word\nClear all: /glossary clear",
        "gl_list": "Your saved terms", "gl_none": "The glossary is empty.", "gl_added": "✅ Added", "gl_deleted": "🗑 Deleted", "gl_cleared": "🗑 Glossary cleared",
        "card_title": "Done", "card_from": "From", "card_to": "To", "card_len": "Clip length",
        "card_time": "Processing time", "card_count": "Subtitles", "card_font": "Font", "card_words": "Words",
        "step_dl": "Downloading the video", "step_audio": "Extracting audio", "step_tr": "Transcribing speech",
        "step_trans": "Translating", "step_burn": "Burning subtitles", "step_up": "Uploading the video", "step_send": "Sending the result",
        "secs": "s",
        "p_dl": "⬇️ Downloading the video...",
        "p_audio": "🎧 Extracting audio...",
        "p_tr_auto": "📝 Transcribing (detecting language)...",
        "p_tr": "📝 Transcribing...",
        "p_trans": "🌐 Translating...",
        "p_burn": "🎬 Burning subtitles into the video...",
        "p_up": "📤 Uploading the video...",
        "caption_burn": "✅ Subtitles added.",
        "caption_txt": "📄 Translation",
        "caption_srt": "Timed subtitles file (SRT)",
        "too_big": "⚠️ The resulting video is over 50MB (Telegram's bot limit), here is the subtitle file instead.",
        "no_speech": "I couldn't detect clear speech in the clip.",
        "too_long": "Maximum clip length in this mode is {m} minutes. Try the text + SRT mode for longer clips.",
        "dl_fail": "Couldn't download the video from this link (it may be private or blocked from the server).",
        "dl_empty": "No video found at this link.",
        "file_big": "Telegram doesn't let bots download files over 20MB. Upload the video somewhere and send me the link instead.",
        "translate_fail": "Translation is unavailable right now. Please try again shortly.",
        "timeout": "Processing took longer than allowed, try a shorter clip.",
        "generic": "An unexpected error occurred while processing. Please try again.",
    },
}


def T(ui: str, key: str, **kw) -> str:
    s = STR.get(ui, STR["en"]).get(key) or STR["en"][key]
    return s.format(**kw) if kw else s


# ----------------------------------------------------------------- timing
def _read_mb(paths):
    for p in paths:
        try:
            with open(p) as f:
                return f"{int(f.read().strip()) / 1048576:.0f}MB"
        except Exception:
            continue
    return "n/a"


def mem_now() -> str:
    return _read_mb(("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"))


def mem_peak() -> str:
    return _read_mb(("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"))


class Stages:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = self.t0 = time.monotonic()
        self.items = []

    def mark(self, name: str):
        now = time.monotonic()
        self.items.append(f"{name}={now - self.t:.1f}s@{mem_now()}")
        self.t = now

    def summary(self) -> str:
        return " ".join(self.items) + f" total={time.monotonic() - self.t0:.1f}s peak={mem_peak()}"


# ---------------------------------------------------------------- whisper
_model = None
_model_lock = threading.Lock()


def get_model() -> WhisperModel:
    global _model
    with _model_lock:
        if _model is None:
            log.info("Loading Whisper model: %s", WHISPER_MODEL)
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS, num_workers=1)
        return _model


def _groq_asr(audio_path: str, model: str, language, detailed: bool, vocab: str = ""):
    with open(audio_path, "rb") as f:
        data = {"model": model, "response_format": "verbose_json", "temperature": "0"}
        if detailed:
            data["timestamp_granularities[]"] = ["word", "segment"]
        if language:
            data["language"] = language
        hint = " ".join(x for x in (GROQ_PROMPT, vocab) if x)
        if hint:
            data["prompt"] = hint[:800]
        r = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            data=data, files={"file": ("audio.flac", f, "audio/flac")}, timeout=120,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def transcribe_groq(audio_path: str, language=None, vocab: str = ""):
    """Returns (segments, language, second_opinion_text). The second model is used later to recover words the first one dropped."""
    d, last = None, None
    for model in (GROQ_ASR_MODEL, GROQ_ASR_FALLBACK):
        try:
            d = _groq_asr(audio_path, model, language, detailed=True, vocab=vocab)
            used = model
            break
        except Exception as e:
            last = e
            log.warning("Groq ASR %s failed: %s", model, str(e)[:200])
    if d is None:
        raise last
    alt = ""
    other = GROQ_ASR_FALLBACK if used != GROQ_ASR_FALLBACK else GROQ_ASR_MODEL
    if other != used:
        try:
            alt = _groq_asr(audio_path, other, language or LANG_NAME_TO_CODE.get(str(d.get("language", "")).lower()), detailed=False, vocab=vocab).get("text", "").strip()
        except Exception as e:
            log.warning("Groq second-opinion ASR failed: %s", str(e)[:150])
    words = [(float(w["start"]), float(w["end"])) for w in d.get("words", []) if "start" in w]
    out = []
    for x in d.get("segments", []):
        text = x.get("text", "").strip()
        if not text or float(x.get("no_speech_prob", 0)) >= 0.8:
            continue
        a, b = float(x["start"]), float(x["end"])
        inside = [w for w in words if a - 0.05 <= (w[0] + w[1]) / 2 <= b + 0.05]
        if inside:  # word timing is tighter than the segment edges -> subtitle matches the voice
            a, b = inside[0][0], inside[-1][1]
        out.append((a + SUB_DELAY, b + SUB_DELAY, text))
    lang = language or LANG_NAME_TO_CODE.get(str(d.get("language", "")).lower(), str(d.get("language", ""))[:2])
    log.info("second opinion (%s): %s", other, alt[:400])
    return out, lang, alt


def transcribe(audio_path: str, language=None, vocab: str = ""):
    """language=None -> auto-detect; otherwise skip detection (faster)."""
    if GROQ_API_KEY:
        try:
            return transcribe_groq(audio_path, language, vocab)
        except Exception as e:
            log.warning("Groq transcription failed, using local Whisper: %s", str(e)[:150])
    segments, info = get_model().transcribe(
        audio_path,
        beam_size=1,
        vad_filter=USE_VAD,
        condition_on_previous_text=False,
        language=language,
    )
    out = [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
    return out, info.language, ""


# -------------------------------------------------------------- translate
MAX_CHUNK = 450  # chars per request (MyMemory limit is ~500 bytes)
_cooldown: dict = {}  # provider name -> unix time until which it is skipped


def _tr_official(text: str, src: str, tgt: str) -> str:
    r = httpx.post(
        "https://translation.googleapis.com/language/translate/v2",
        params={"key": GOOGLE_API_KEY},
        data={"q": text, "target": tgt, "format": "text"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["data"]["translations"][0]["translatedText"]


def _tr_gtx(text: str, src: str, tgt: str) -> str:
    r = httpx.post(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": "auto", "tl": tgt, "dt": "t"},
        data={"q": text},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    r.raise_for_status()
    return "".join(part[0] for part in r.json()[0] if part and part[0])


def _tr_mymemory(text: str, src: str, tgt: str) -> str:
    r = httpx.get(
        "https://api.mymemory.translated.net/get",
        params={"q": text, "langpair": f"{src}|{tgt}"},
        timeout=20,
    )
    r.raise_for_status()
    d = r.json()
    out = (d.get("responseData") or {}).get("translatedText") or ""
    if str(d.get("responseStatus")) != "200" or "MYMEMORY WARNING" in out:
        raise RuntimeError(str(d.get("responseDetails") or out)[:100])
    return out


def _translate_text(text: str, src: str, tgt: str) -> str:
    """Try providers in order; a provider that keeps failing is skipped for 10 minutes."""
    providers = []
    if GOOGLE_API_KEY:
        providers.append(("google_api", _tr_official))
    providers += [("gtx", _tr_gtx), ("mymemory", _tr_mymemory)]
    for name, fn in providers:
        if _cooldown.get(name, 0) > time.time():
            continue
        for attempt in range(2):
            try:
                res = fn(text, src, tgt)
                if res and res.strip():
                    return res
            except Exception as e:
                log.warning("translate via %s failed (try %d): %s", name, attempt + 1, str(e)[:120])
                time.sleep(1)
        _cooldown[name] = time.time() + 600
    raise RuntimeError("all translation providers failed")


LANG_NAME_TO_CODE = {
    "arabic": "ar", "english": "en", "french": "fr", "spanish": "es", "german": "de", "turkish": "tr",
    "persian": "fa", "urdu": "ur", "hindi": "hi", "russian": "ru", "italian": "it", "portuguese": "pt",
    "indonesian": "id", "chinese": "zh", "japanese": "ja", "korean": "ko", "dutch": "nl", "polish": "pl",
    "bengali": "bn", "swahili": "sw", "hebrew": "he", "ukrainian": "uk", "greek": "el", "thai": "th",
    "vietnamese": "vi", "malay": "ms", "swedish": "sv", "romanian": "ro", "czech": "cs", "somali": "so",
}


_LLM_MODELS = [m for m in dict.fromkeys([GROQ_LLM_MODEL, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]) if m]
_llm_bad: dict = {}   # model -> time until which it is skipped


def _llm_batch(texts, src_name, tgt_name, fix_only, alt="", gloss=None):
    task = (
        f"These are consecutive subtitle lines from a video, auto-transcribed from speech ({src_name}). "
        "The transcription may contain mistakes (wrong or misheard words, names, missing letters). "
        "NEVER delete or skip words or sentences: every spoken word must stay in the output. "
        "If a word looks garbled, replace it with the most likely intended word given the context. "
    )
    if alt:
        task += ("A second speech-recognition model transcribed the same audio as: \"" + alt[:1500] + "\". "
                 "Where the lines miss a word or phrase that the second transcript has and it fits the context, "
                 "put it back into the right line. ")
    if gloss:
        task += ("User glossary - always use exactly these terms/spellings (format: heard => correct): "
                 + "; ".join(f"{a} => {b}" if a != b else a for a, b in list(gloss.items())[:40]) + ". ")
    if fix_only:
        task += "Fix obvious transcription mistakes using the context. Keep the same language and meaning."
    else:
        task += (f"First fix obvious mistakes using the context, then translate each line into natural, "
                 f"fluent {tgt_name}, as a native speaker would say it. Keep proper names correct.")
    task += (' Return JSON {"t": [...]} with EXACTLY the same number of items, in the same order, '
             "one output string per input line. No explanations.")
    last = None
    for model in _LLM_MODELS:
        if _llm_bad.get(model, 0) > time.time():
            continue
        body = {"model": model, "temperature": 0.2, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": task},
                             {"role": "user", "content": json.dumps({"lines": texts}, ensure_ascii=False)}]}
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
        try:
            r = httpx.post("https://api.groq.com/openai/v1/chat/completions",
                           headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json=body, timeout=60)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            out = json.loads(r.json()["choices"][0]["message"]["content"])["t"]
            if not isinstance(out, list) or len(out) != len(texts):
                raise ValueError("line count mismatch")
            log.info("LLM used: %s", model)
            return [str(x).strip() for x in out]
        except Exception as e:
            last = e
            log.warning("Groq LLM %s failed: %s", model, str(e)[:250])
            if "HTTP 404" in str(e) or "HTTP 403" in str(e):
                _llm_bad[model] = time.time() + 3600   # not available for this account -> skip for an hour
    raise last or RuntimeError("no LLM model available")


def translate_segments_llm(segments, src: str, tgt: str, alt: str = "", gloss=None):
    """Context-aware translation + correction via Groq. Raises on failure."""
    names = {c: n for n, c in LANG_NAME_TO_CODE.items()}
    src_name, tgt_name = names.get(src, src), names.get(tgt, tgt)
    texts = [" ".join(t.split()) for _, _, t in segments]
    result = []
    for i in range(0, len(texts), 30):
        batch = texts[i:i + 30]
        for attempt in range(2):
            try:
                result += _llm_batch(batch, src_name, tgt_name, fix_only=(src == tgt), alt=alt if len(texts) <= 30 else "", gloss=gloss)
                break
            except Exception:
                if attempt == 1:
                    raise
                time.sleep(2)
    return [(s, e, r or t) for (s, e, _), r, t in zip(segments, result, texts)]


def translate_segments(segments, src: str, tgt: str, alt: str = "", gloss=None):
    """Translate in a few small batches (one segment per line)."""
    if GROQ_API_KEY:
        try:
            return translate_segments_llm(segments, src, tgt, alt, gloss)
        except Exception as e:
            log.warning("Groq translation failed, using fallback chain: %s", str(e)[:150])
    if src == tgt:
        return segments
    texts = [" ".join(t.split()) for _, _, t in segments]

    chunks, cur, size = [], [], 0
    for i, t in enumerate(texts):
        if cur and size + len(t) + 1 > MAX_CHUNK:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(t) + 1
    if cur:
        chunks.append(cur)

    result = list(texts)
    ok = 0
    for idxs in chunks:
        joined = "\n".join(texts[i] for i in idxs)
        try:
            out = _translate_text(joined, src, tgt).split("\n")
        except Exception:
            log.warning("chunk translation failed; keeping original text for it")
            continue
        if len(out) == len(idxs):
            for i, line in zip(idxs, out):
                result[i] = line.strip() or texts[i]
            ok += len(idxs)
            time.sleep(0.5)
            continue
        log.warning("line count mismatch (%d vs %d); translating one by one", len(out), len(idxs))
        for i in idxs:
            try:
                result[i] = _translate_text(texts[i], src, tgt)
                ok += 1
            except Exception:
                pass
            time.sleep(0.4)

    if ok == 0:
        raise UserError("translate_fail")
    return [(s, e, r) for (s, e, _), r in zip(segments, result)]


# --------------------------------------------------- text clean / wrapping
# invisible direction/format characters that show up as empty boxes in burned subtitles
_STRIP = (
    list(range(0x200B, 0x2010))      # ZWSP, ZWNJ, ZWJ, LRM, RLM
    + list(range(0x202A, 0x202F))    # embeddings / overrides
    + list(range(0x2060, 0x2070))    # word joiner, isolates, invisible operators
    + [0xFEFF, 0x00AD, 0x061C]       # BOM, soft hyphen, Arabic letter mark
)
_STRIP_MAP = dict.fromkeys(_STRIP, None)
_KEEP_JOINERS = {0x200C, 0x200D}  # meaningful in Persian/Urdu spelling


def clean_text(text: str, keep_joiners: bool = False) -> str:
    table = {k: None for k in _STRIP if not (keep_joiners and k in _KEEP_JOINERS)}
    return " ".join(text.translate(table).split())


_AR_MARKS = (
    set(range(0x064B, 0x0660)) | {0x0670, 0x0640}      # tashkeel, superscript alef, tatweel
    | set(range(0x06D6, 0x06EE))                        # Quranic annotation marks
)
_AR_MARKS_MAP = {c: None for c in _AR_MARKS}


def normalize_ar(text: str) -> str:
    """Arabic-script text: NFKC (presentation forms -> letters), drop diacritics, alef wasla -> alef."""
    text = unicodedata.normalize("NFKC", text).translate(_AR_MARKS_MAP)
    return text.replace("\u0671", "\u0627")


def odd_chars(text: str) -> str:
    """Characters outside plain letters/digits/punctuation - helps find what renders as a box."""
    seen = {}
    for ch in text:
        o = ord(ch)
        if ch in " \n" or 0x20 <= o <= 0x7E or 0x0621 <= o <= 0x064A or o in (0x060C, 0x061B, 0x061F, 0x066A, 0x066B, 0x066C):
            continue
        if 0x0660 <= o <= 0x0669 or 0x06F0 <= o <= 0x06F9:
            continue
        seen[ch] = f"U+{o:04X} {unicodedata.name(ch, '?')}"
    return "; ".join(list(seen.values())[:12])


def wrap2(line: str, max_chars: int) -> str:
    """Wrap into at most two balanced lines."""
    if len(line) <= max_chars:
        return line
    words = line.split()
    best, best_diff = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        diff = abs(len(a) - len(b))
        if best_diff is None or diff < best_diff:
            best, best_diff = (a, b), diff
    return f"{best[0]}\n{best[1]}" if best else line


def make_cues(segments, max_chars: int, origs=None):
    """Split long segments into several timed cues of at most two short lines.
    origs (optional): original-language text per segment, added as small marked lines (\\x01) under the translation."""
    cues = []
    for idx, (start, end, text) in enumerate(segments):
        words = text.split()
        if not words:
            continue
        ow = origs[idx].split() if origs and origs[idx] else []
        chunks, cur, n = [], [], 0
        for w in words:
            add = len(w) + (1 if cur else 0)
            if cur and n + add > 2 * max_chars:
                chunks.append(cur)
                cur, n = [w], len(w)
            else:
                cur.append(w)
                n += add
        if cur:
            chunks.append(cur)
        total = sum(len(" ".join(c)) for c in chunks) or 1
        t, dur, cum = start, end - start, 0
        for c in chunks:
            line = " ".join(c)
            d = dur * len(line) / total
            txt = wrap2(line, max_chars)
            if ow:
                lo = round(len(ow) * cum / total)
                cum += len(line)
                hi = round(len(ow) * cum / total)
                oline = " ".join(ow[lo:hi])
                if oline:
                    txt += "\n" + "\n".join("\x01" + x for x in wrap2(oline, int(max_chars * 1.4)).split("\n"))
            cues.append((t, t + d, txt))
            t += d
    return cues


def fix_overlaps(cues, gap=0.04, min_show=0.35):
    """Cues must never be on screen together: each one ends before the next starts, and is held long
    enough to read (with a small hold after the voice, because word timings are only approximate)."""
    cues = sorted(([a, b, t] for a, b, t in cues), key=lambda c: c[0])
    for i, c in enumerate(cues):
        a, b, t = c
        want = max(b + 0.25, a + max(0.8, 0.045 * len(t)))   # hold: at least ~0.8 s or 45 ms per character
        if i + 1 < len(cues):
            limit = cues[i + 1][0] - gap
            if limit - a >= min_show:
                want = min(want, limit)
            else:                                           # next one starts too soon: show this one briefly, delay the next
                want = a + min_show
                cues[i + 1][0] = want + gap
                cues[i + 1][1] = max(cues[i + 1][1], cues[i + 1][0] + min_show)
        c[1] = max(want, a + 0.1)
    return [(a, b, t) for a, b, t in cues]


# -------------------------------------------------------------------- srt
def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(cues, keep_marks: bool = False) -> str:
    blocks = []
    for i, (start, end, text) in enumerate(cues, start=1):
        if not keep_marks:
            text = text.replace("\x01", "")
        blocks.append(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
    return "\n".join(blocks)


# ----------------------------------------------------------------- ffmpeg
async def run_cmd(args, cwd=None, timeout=900, grep=()) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise UserError("timeout")
    if proc.returncode != 0:
        tail = err.decode(errors="ignore")[-1000:]
        raise RuntimeError(f"{args[0]} exited {proc.returncode}: {tail}")
    if grep:
        seen = []
        for line in err.decode(errors="ignore").splitlines():
            if any(g in line for g in grep) and line not in seen:
                seen.append(line)
        for line in seen[:15]:
            log.info("ffmpeg: %s", line[:220])
    return out


async def probe(video: Path):
    """Return (width, height, duration_seconds) with rotation taken into account."""
    raw = await run_cmd(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation:format=duration",
            "-of", "json", str(video),
        ],
        timeout=60,
    )
    info = json.loads(raw or b"{}")
    stream = (info.get("streams") or [{}])[0]
    w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
    try:  # non-square pixels: the real display width is width * SAR
        sa, _, sb = str(stream.get("sample_aspect_ratio") or "1:1").partition(":")
        sar = float(sa) / float(sb or 1)
        if w and 0.2 < sar < 5 and abs(sar - 1) > 0.01:
            w = int(w * sar)
    except (ValueError, ZeroDivisionError):
        pass
    rot = (stream.get("tags") or {}).get("rotate")
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    if rot is not None and abs(int(float(rot))) % 180 == 90:
        w, h = h, w
    duration = float((info.get("format") or {}).get("duration") or 0)
    return w, h, duration


def out_size(w: int, h: int):
    """Size of the burned video: short side = MAX_RES (same rule burn_subtitles uses)."""
    short = min(w, h)
    if short and short != MAX_RES:
        f = MAX_RES / short
        return int(w * f) // 2 * 2, int(h * f) // 2 * 2
    return w // 2 * 2, h // 2 * 2


def chars_per_line(w: int, h: int, job: dict) -> int:
    """How many characters of the chosen font fit on one subtitle line, so lines never overflow the frame."""
    ow, oh = out_size(w, h)
    if not (ow and oh):
        return 28
    tgt = job["tgt"]
    fonts = fonts_for(tgt)
    font = fonts[job["font_i"] % len(fonts)]
    group = "ar" if tgt in ARABIC_SCRIPT else "lat"
    base = 12 if oh > ow else 21
    px = base * SIZES[job["size_i"]][1] * font_scale(font, group) * oh / 288
    avail = ow * (1 - 40 / 384)                     # side margins are 20 of 384 units each
    if tgt in ARABIC_SCRIPT:
        factor = 0.46
    elif tgt in CJK:
        factor = 1.0
    else:
        factor = 0.68 if font.startswith("DejaVu") else 0.6
    return max(10, int(avail / (px * factor)))


async def src_fps(video: Path) -> float:
    try:
        raw = await run_cmd(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=avg_frame_rate", "-of", "csv=p=0", str(video)], timeout=30)
        num, _, den = raw.decode().strip().partition("/")
        return float(num) / float(den or 1)
    except Exception:
        return 0.0


async def burn_subtitles(video: Path, srt_name: str, workdir: Path, w: int, h: int, job: dict,
                         duration: float = 0.0, on_frac=None) -> Path:
    w, h = out_size(w, h)
    scale = f"scale={w}:{h},setsar=1," if w and h else ""   # exact size + square pixels = same shape as the original
    portrait = h > w

    fonts = fonts_for(job["tgt"])
    font = fonts[job["font_i"] % len(fonts)]
    log.info("burn: font=%s size=x%s style=%s pos=%s res=%sx%s", font, SIZES[job["size_i"]][1],
             STYLES[job["style_i"]], POSITIONS[job["pos_i"]][0], w, h)
    mult = SIZES[job["size_i"]][1]
    base = 12 if portrait else 21  # relative to a 288px-high canvas
    font_size = max(6, round(base * mult * font_scale(font, "ar" if job["tgt"] in ARABIC_SCRIPT else "lat")))
    style_key = STYLES[job["style_i"]]
    align = POSITIONS[job["pos_i"]][1]
    if align == 2:
        margin_v = 22 if portrait else 12
    elif align == 8:
        margin_v = 30 if portrait else 12
    else:
        margin_v = 0

    # force_style's Alignment uses the legacy SSA numbering (5 = top-left!), so the position is set
    # with an \anN tag on every cue instead, which always uses the numpad layout.
    style = (
        f"FontName={font},Bold=1,FontSize={font_size},{STYLE_DEFS[style_key]},"
        f"Alignment=2,MarginV={margin_v},MarginL=20,MarginR=20"
    )
    src_srt = (workdir / srt_name).read_text(encoding="utf-8")
    tagged = []
    for block in src_srt.strip().split("\n\n"):
        lines = block.split("\n")
        if len(lines) >= 3:
            fixed = []
            for ln in lines[2:]:
                small = ln.startswith("\x01")
                ln = ln.lstrip("\x01")
                rtl = (job.get("lang") if small else job["tgt"]) in RTL_LANGS
                if rtl:   # keeps ". ، ! ( )" at the correct (left) end of right-to-left lines
                    ln += "\u200f"
                if small:  # original-language line: smaller and softer
                    ln = "{\\fs%d\\1c&H00D9D9D9&}" % max(5, round(font_size * 0.7)) + ln
                fixed.append(ln)
            lines[2:] = fixed
            lines[2] = "{\\an%d}" % align + lines[2]
        tagged.append("\n".join(lines))
    srt_name = "subs_burn.srt"
    (workdir / srt_name).write_text("\n\n".join(tagged) + "\n", encoding="utf-8")
    out = workdir / "output.mp4"
    # srt is referenced by relative name and ffmpeg runs inside workdir -> no path escaping issues
    w, h = w // 2 * 2, h // 2 * 2
    fps_cap = "fps=30," if await src_fps(video) > 30.5 else ""   # 50/60 fps -> 30 fps: half the work, same look for subtitles
    scale = fps_cap + scale
    if SS > 1:
        vf = (f"{scale}scale={w * SS}:{h * SS}:flags=bicubic,subtitles={srt_name}:force_style='{style}',"
              f"scale={w}:{h}:flags=area")
    else:
        vf = f"{scale}subtitles={srt_name}:force_style='{style}'"
    await run_ffmpeg(
        [
            "ffmpeg", "-y", "-v", "verbose", "-threads", "1", "-filter_threads", "1", "-i", str(video), "-vf", vf,
            "-c:v", "libx264", "-preset", BURN_PRESET, "-crf", "29", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-threads", "1", "-movflags", "+faststart", str(out),
        ],
        cwd=str(workdir),
        timeout=1500,
        duration=duration,
        on_frac=on_frac,
        grep=("fontselect", "Glyph"),
    )
    return out


# --------------------------------------------------------------- download
def _ytdlp_download(url: str, workdir: Path, audio_only: bool = False) -> Path:
    opts = {
        "outtmpl": str(workdir / "input.%(ext)s"),
        # text-only mode needs just the audio -> much smaller/faster download
        "format": "ba/b" if audio_only else "bv*+ba/b",
        "format_sort": [f"res:{MAX_RES}", "ext:mp4:m4a"],
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
        "max_filesize": 300 * 1024 * 1024,
    }
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        opts["cookiefile"] = COOKIES_FILE
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        log.warning("yt-dlp failed for %s: %s", url, e)
        raise UserError("dl_fail")
    files = [p for p in workdir.glob("input.*") if p.suffix not in (".part", ".ytdl")]
    if not files:
        raise UserError("dl_empty")
    return files[0]


async def fetch_input(bot, kind: str, data: str, workdir: Path, audio_only: bool = False) -> Path:
    if kind == "url":
        return await asyncio.to_thread(_ytdlp_download, data, workdir, audio_only)
    dest = workdir / "input.mp4"
    try:
        tg_file = await bot.get_file(data)
        await tg_file.download_to_drive(dest)
    except BadRequest as e:
        if "too big" in str(e).lower():
            raise UserError("file_big")
        raise
    return dest


# --------------------------------------------------------------- pipeline
async def set_status(msg, text: str):
    try:
        await msg.edit_text(text)
    except TelegramError:
        pass


LOGO_B64 = """
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
"""   # round bot logo (280px JPEG), used in the welcome picture and result cards


# ------------------------------------------------- saved user settings + glossary
PREFS_FILE = Path(os.environ.get("DATA_DIR", tempfile.gettempdir())) / "subvid_prefs.json"
PREFS: dict = {}          # str(user id) -> {"ui","tgt","mode","font_i","size_i","style_i","pos_i","bi","gl":{wrong: right}}
_prefs_lock = threading.Lock()


def load_prefs():
    try:
        PREFS.update(json.loads(PREFS_FILE.read_text(encoding="utf-8")))
        log.info("prefs loaded for %d users", len(PREFS))
    except Exception:
        pass


def save_prefs():
    try:
        with _prefs_lock:
            tmp = PREFS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(PREFS, ensure_ascii=False), encoding="utf-8")
            tmp.replace(PREFS_FILE)
    except Exception as e:
        log.warning("could not save prefs: %s", e)


def uprefs(uid) -> dict:
    return PREFS.setdefault(str(uid), {})


def apply_glossary(segments, gl: dict):
    """Replace known wrong spellings with the user's correct term (whole words only)."""
    if not gl:
        return segments
    def rx(a):   # Latin terms: whole words; Arabic-script terms: may carry prefixes like بـ/ل/و/ال
        e = re.escape(a)
        return re.compile(e if _has_arabic(a) else r"(?<!\w)" + e + r"(?!\w)", re.I)
    pats = [(rx(a), b) for a, b in gl.items() if a and a != b]
    out = []
    for st, en, t in segments:
        for rx, b in pats:
            t = rx.sub(lambda m, b=b: b, t)
        out.append((st, en, t))
    return out


def parse_glossary(text: str) -> dict:
    out = {}
    for line in text.replace("،", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in ("=>", "->", "=", "←", "→"):
            if sep in line:
                a, b = (x.strip() for x in line.split(sep, 1))
                break
        else:
            a = b = line
        if a and b and len(a) <= 40 and len(b) <= 40:
            out[a] = b
    return out


_alert_last: dict = {}


async def notify_admin(bot, text: str, key: str = ""):
    """Short alert to the owner; the same alert is sent at most once a minute."""
    if not ADMIN_ID:
        return
    key = key or text[:80]
    now = time.monotonic()
    if now - _alert_last.get(key, -999) < 60:
        return
    _alert_last[key] = now
    try:
        await bot.send_message(ADMIN_ID, text[:3500])
    except TelegramError:
        pass


def install_health_route():
    """Adds GET/HEAD /health -> 200 'ok' to the webhook server so uptime monitors see the bot as up."""
    try:
        import tornado.web
        from telegram.ext._utils import webhookhandler as wh

        class Health(tornado.web.RequestHandler):
            def get(self):
                self.set_header("Content-Type", "text/plain")
                self.write("ok")

            def head(self):
                self.set_status(200)

        original = wh.WebhookAppClass.__init__

        def patched(self, *a, **k):
            original(self, *a, **k)
            self.add_handlers(r".*", [(r"/(?:health)?/?", Health)])

        wh.WebhookAppClass.__init__ = patched
        log.info("health endpoint enabled (GET /health)")
    except Exception as e:
        log.warning("could not add the health endpoint: %s", e)


_BANNER: dict = {}   # "file_id" or "png"


def _esc(t) -> str:
    return str(t).replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _has_arabic(t: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in t)


def _fmt_time(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}"


async def _render_card(ass: str, logo_size: int, logo_x: int, logo_y: int, W: int = 1280, H: int = 720):
    """gradient background + round logo + ASS text -> PNG bytes (or None)."""
    workdir = Path(tempfile.mkdtemp(prefix="card_"))
    try:
        (workdir / "c.ass").write_text(ass, encoding="utf-8")
        (workdir / "logo.jpg").write_bytes(base64.b64decode("".join(LOGO_B64.split())))
        fc = (f"[1:v]scale={logo_size}:{logo_size},format=gbrap,"
              "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lt(hypot(X-W/2,Y-H/2),W/2-1),255,0)'[lg];"
              f"[0:v][lg]overlay={logo_x}:{logo_y},ass=c.ass")
        for bg in (f"gradients=s={W}x{H}:c0=0x050a14:c1=0x0c3558:x0=0:y0=0:x1={W}:y1={H}:d=1",
                   f"color=c=0x081a2c:s={W}x{H}:d=1"):
            rc, _, err = await run_capture(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", bg, "-i", "logo.jpg",
                                            "-filter_complex", fc, "-frames:v", "1", "c.png"], cwd=str(workdir))
            if rc == 0 and (workdir / "c.png").exists():
                return (workdir / "c.png").read_bytes()
        log.warning("card render failed: %s", err.decode(errors="ignore")[-200:])
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


_ASS_HEAD = (
    "[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n\n"
    "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
    "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding\n"
    "Style: T,Noto Sans,170,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,3,0,1,0,0,5,0,0,0,1\n"
    "Style: A,Noto Naskh Arabic,76,&H00F2D34A,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1\n"
    "Style: P,Noto Naskh Arabic,62,&H00FFFFFF,&H000000FF,&H00100A06,&H00100A06,1,0,0,0,100,100,0,0,3,20,0,5,0,0,0,1\n"
    "Style: D,Noto Sans,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
    "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)


async def banner_png():
    """Welcome picture (16:9): the bot's own logo, big simple text, bot colours."""
    if "png" in _BANNER:
        return _BANNER["png"]
    ass = _ASS_HEAD + (
        "Dialogue: 0,0:00:00.00,0:00:05.00,T,,0,0,0,,{\\pos(640,340)}SubVid Ai\n"
        "Dialogue: 0,0:00:00.00,0:00:05.00,A,,0,0,0,,{\\pos(640,465)}ترجم فيديوهاتك في ثوانٍ\n"
        "Dialogue: 1,0:00:00.00,0:00:05.00,P,,0,0,0,,{\\pos(640,600)}السلام عليكم ورحمة الله\n"
    )
    png = await _render_card(ass, 230, 525, 22)
    if png:
        _BANNER["png"] = png
    return png


async def result_card(ui: str, rows: list):
    """Closing card: logo, 'Done', and 6 label/value cells. rows = [(label, value), ...] (max 6)."""
    rtl = ui == "ar"
    label_font = "Noto Naskh Arabic" if rtl else "Noto Sans"
    panel = "m 100 300 l 1180 300 l 1180 690 l 100 690"
    ev = [
        f"Dialogue: 0,0:00:00.00,0:00:05.00,D,,0,0,0,,{{\\an7\\pos(0,0)\\p1\\bord0\\1c&H00281406&\\1a&H40&}}{panel}{{\\p0}}",
        f"Dialogue: 1,0:00:00.00,0:00:05.00,A,,0,0,0,,{{\\pos(640,235)\\1c&H0099D352&\\fs84\\fn{label_font}}}✓ {_esc(T(ui, 'card_title'))}",
    ]
    xs = [1130, 600] if rtl else [150, 680]
    an = 6 if rtl else 4
    for i, (label, value) in enumerate(rows[:6]):
        col, row = i % 2, i // 2
        y = 330 + row * 120
        vfont = "Noto Naskh Arabic" if _has_arabic(str(value)) else "Noto Sans"
        ev.append(f"Dialogue: 1,0:00:00.00,0:00:05.00,D,,0,0,0,,{{\\an{an}\\pos({xs[col]},{y})\\fs32\\1c&H00C9B79A&\\fn{label_font}}}{_esc(label)}")
        ev.append(f"Dialogue: 1,0:00:00.00,0:00:05.00,D,,0,0,0,,{{\\an{an}\\pos({xs[col]},{y + 55})\\fs48\\b1\\fn{vfont}}}{_esc(value)}")
    return await _render_card(_ASS_HEAD + "\n".join(ev) + "\n", 170, 555, 18)


async def send_banner(msg):
    try:
        photo = _BANNER.get("file_id") or await banner_png()
        if not photo:
            return
        sent = await msg.reply_photo(photo=photo)
        if sent.photo:
            _BANNER["file_id"] = sent.photo[-1].file_id   # next time Telegram reuses it: no upload
    except TelegramError as e:
        log.warning("banner not sent: %s", e)


async def send_result_card(bot, chat_id, ui, rows):
    """Never lets a decoration problem break a finished job."""
    try:
        png = await result_card(ui, rows)
        if png:
            await bot.send_photo(chat_id, photo=png)
    except Exception as e:
        log.warning("result card not sent: %s", e)


class Progress:
    """One status message that shows a checklist (and a bar while burning). Edits are throttled."""

    def __init__(self, ui, msg, mode):
        self.ui, self.msg = ui, msg
        self.steps = (["dl", "audio", "tr", "trans", "burn", "up"] if mode == "sub"
                      else ["dl", "audio", "tr", "trans", "send"])
        self.cur, self.frac, self.last = None, None, 0.0

    def _text(self) -> str:
        lines = [T(self.ui, "proc_title"), ""]
        idx = self.steps.index(self.cur) if self.cur in self.steps else -1
        for i, k in enumerate(self.steps):
            label = T(self.ui, "step_" + k)
            if i < idx:
                lines.append(f"✅ {label}")
            elif i == idx:
                bar = ""
                if self.frac is not None:
                    n = int(self.frac * 10)
                    bar = f"  {'▰' * n}{'▱' * (10 - n)} {int(self.frac * 100)}%"
                lines.append(f"🔄 {label}{bar}")
            else:
                lines.append(f"▫️ {label}")
        return "\n".join(lines)

    async def step(self, key: str):
        self.cur, self.frac, self.last = key, None, time.monotonic()
        await set_status(self.msg, self._text())

    async def update(self, frac: float):
        if time.monotonic() - self.last < 4:   # Telegram edit limits + keeps the server light
            return
        self.frac, self.last = max(0.0, min(frac, 0.99)), time.monotonic()
        await set_status(self.msg, self._text())


async def run_ffmpeg(args, cwd=None, timeout=1500, duration=0.0, on_frac=None, grep=()):
    """ffmpeg with live progress (-progress pipe:1); stderr is read line by line so memory stays small."""
    args = [args[0], "-progress", "pipe:1", "-nostats"] + list(args[1:])
    proc = await asyncio.create_subprocess_exec(
        *args, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    tail, seen = deque(maxlen=25), []

    async def read_err():
        async for raw in proc.stderr:
            line = raw.decode(errors="ignore").rstrip()
            tail.append(line)
            if grep and len(seen) < 15 and any(g in line for g in grep) and line not in seen:
                seen.append(line)

    async def read_out():
        async for raw in proc.stdout:
            line = raw.decode(errors="ignore").strip()
            if line.startswith("out_time_us=") and duration and on_frac:
                try:
                    await on_frac(int(line.split("=", 1)[1]) / 1e6 / duration)
                except ValueError:
                    pass

    try:
        await asyncio.wait_for(asyncio.gather(read_out(), read_err(), proc.wait()), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise UserError("timeout")
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} exited {proc.returncode}: " + "\n".join(list(tail)[-8:])[-1000:])
    for line in seen:
        log.info("ffmpeg: %s", line[:220])


async def process(bot, chat_id: int, job: dict, status):
    ui, mode, tgt = job["ui"], job["mode"], job["tgt"]
    workdir = Path(tempfile.mkdtemp(prefix="subvid_"))
    st = Stages()
    prog = Progress(ui, status, mode)
    PENDING.append(job["id"])
    try:
        ahead = PENDING.index(job["id"])
        if ahead > 0:
            await set_status(status, T(ui, "queue", n=ahead))
        async with JOB_SEM:
            st.reset()  # don't count the time spent waiting in the queue
            await prog.step("dl")
            video = await fetch_input(bot, job["kind"], job["data"], workdir, audio_only=(mode == "txt"))
            st.mark("download")

            w, h, duration = await probe(video)
            st.mark("probe")
            limit = MAX_MINUTES_TXT if mode == "txt" else MAX_MINUTES
            if duration and duration > limit * 60:
                raise UserError("too_long", m=limit)

            await prog.step("audio")
            # clean the audio (rumble/hiss, uneven volume) so recognition doesn't depend on recording quality
            audio = workdir / "audio.flac"
            await run_cmd(
                ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                 "-af", "highpass=f=80,lowpass=f=7600,afftdn=nf=-28,dynaudnorm=f=150:g=15",
                 str(audio)]
            )
            st.mark("audio")
            if audio.stat().st_size > 24 * 1024 * 1024:
                raise UserError("too_long", m=max(1, int(limit * 24 * 1024 * 1024 / audio.stat().st_size)))

            await prog.step("tr")
            gl = PREFS.get(str(job.get("uid")), {}).get("gl", {}) if job.get("uid") else {}
            vocab = ", ".join(dict.fromkeys(gl.values()))[:500]
            segments, lang, alt = await asyncio.to_thread(transcribe, str(audio), job["src"], vocab)
            st.mark("transcribe")
            log.info("raw transcript (%s): %s", lang, " | ".join(t for _, _, t in segments)[:600])
            if not segments:
                raise UserError("no_speech")

            segments = apply_glossary(segments, gl)
            job["lang"] = lang
            orig_texts = [" ".join(t.split()) for _, _, t in segments]
            if lang != tgt or GROQ_API_KEY:
                await prog.step("trans")
                segments = await asyncio.to_thread(translate_segments, segments, lang, tgt, alt, gl)
                st.mark("translate")

            keep_joiners = tgt in ("fa", "ur")
            segments = [(s, e, clean_text(t, keep_joiners)) for s, e, t in segments]
            if tgt in ARABIC_SCRIPT:
                raw_odd = odd_chars(" ".join(t for _, _, t in segments))
                segments = [(s, e, normalize_ar(t)) for s, e, t in segments]
                log.info("subtitle text: odd chars before=[%s] after=[%s]", raw_odd,
                         odd_chars(" ".join(t for _, _, t in segments)))
            origs = orig_texts if (job.get("bi") and mode == "sub" and lang != tgt and len(orig_texts) == len(segments)) else None
            if origs:
                keep = [i for i, x in enumerate(segments) if x[2]]
                origs = [origs[i] for i in keep]
            segments = [x for x in segments if x[2]]
            if not segments:
                raise UserError("no_speech")

            portrait = bool(w and h and h > w)
            mult = SIZES[job["size_i"]][1]
            max_chars = chars_per_line(w, h, job) if mode == "sub" else 40
            cues = fix_overlaps(make_cues(segments, max_chars, origs))
            srt_text = build_srt(cues)
            (workdir / "subs.srt").write_text(srt_text, encoding="utf-8")
            (workdir / "subs_tagged.srt").write_text(build_srt(cues, keep_marks=True), encoding="utf-8")
            plain = "\n".join(t for _, _, t in segments)

            if mode == "txt":
                await prog.step("send")
                if len(plain) <= 3500:
                    await bot.send_message(chat_id, f"{T(ui, 'caption_txt')}:\n\n{plain}")
                else:
                    (workdir / "translation.txt").write_text(plain, encoding="utf-8")
                    with open(workdir / "translation.txt", "rb") as f:
                        await bot.send_document(chat_id, f, caption=T(ui, "caption_txt"))
                with open(workdir / "subs.srt", "rb") as f:
                    await bot.send_document(chat_id, f, filename="subtitles.srt",
                                            caption=T(ui, "caption_srt"))
                st.mark("send")
                await send_result_card(bot, chat_id, ui, [
                    (T(ui, "card_from"), LANGS.get(lang, lang)), (T(ui, "card_to"), LANGS.get(tgt, tgt)),
                    (T(ui, "card_len"), _fmt_time(duration)), (T(ui, "card_time"), _fmt_time(time.monotonic() - st.t0)),
                    (T(ui, "card_count"), len(cues)), (T(ui, "card_words"), sum(len(t.split()) for _, _, t in segments))])
            else:
                await prog.step("burn")
                out = await burn_subtitles(video, "subs_tagged.srt", workdir, w, h, job, duration, prog.update)
                st.mark("burn")
                if out.stat().st_size > TG_UPLOAD_LIMIT:
                    with open(workdir / "subs.srt", "rb") as f:
                        await bot.send_document(chat_id, f, filename="subtitles.srt",
                                                caption=T(ui, "too_big"))
                else:
                    await prog.step("up")
                    elapsed = int(time.monotonic() - st.t0)
                    caption = (f"{T(ui, 'caption_burn')}\n🌐 {LANGS.get(lang, lang)} ➜ {LANGS.get(tgt, tgt)}"
                               f" · ⏱ {elapsed}{T(ui, 'secs')}")
                    thumb = workdir / "thumb.jpg"
                    try:   # small cover picture: Telegram shows it as the preview
                        await run_cmd(["ffmpeg", "-y", "-ss", str(min(1.0, (duration or 2) / 2)), "-i", str(out),
                                       "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "5", str(thumb)], timeout=60)
                    except Exception:
                        thumb = None
                    ow, oh = out_size(w, h)
                    with open(out, "rb") as f:
                        tf = open(thumb, "rb") if thumb and thumb.exists() else None
                        try:
                            await bot.send_video(chat_id, f, supports_streaming=True, caption=caption,
                                                 width=ow or None, height=oh or None, duration=int(duration) or None,
                                                 thumbnail=tf)
                        finally:
                            if tf:
                                tf.close()
                st.mark("upload")
                fonts_used = fonts_for(tgt)
                await send_result_card(bot, chat_id, ui, [
                    (T(ui, "card_from"), LANGS.get(lang, lang)), (T(ui, "card_to"), LANGS.get(tgt, tgt)),
                    (T(ui, "card_len"), _fmt_time(duration)), (T(ui, "card_time"), _fmt_time(time.monotonic() - st.t0)),
                    (T(ui, "card_count"), len(cues)), (T(ui, "card_font"), fonts_used[job["font_i"] % len(fonts_used)])])
        try:
            await status.delete()
        except TelegramError:
            pass
    except UserError as e:
        await set_status(status, "⚠️ " + T(ui, e.key, **e.kw))
    except Exception as e:
        log.exception("job failed")
        await set_status(status, "❌ " + T(ui, "generic"))
        await notify_admin(bot, f"❌ فشل طلب {job['id']}\nالمرحلة: {prog.cur}\n{mode}: {job.get('src') or 'auto'} ← {tgt}\n"
                                f"الخطأ: {type(e).__name__}: {str(e)[:400]}", key=f"job-{type(e).__name__}")
    finally:
        log.info("job %s (%s -> %s, %s) timings: %s", job["id"], job.get("src") or "auto",
                 tgt, mode, st.summary())
        if job["id"] in PENDING:
            PENDING.remove(job["id"])
        shutil.rmtree(workdir, ignore_errors=True)
        gc.collect()


# ---------------------------------------------------------------- screens
def btn(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text, callback_data=data)


def screen_ui_picker():
    rows = [[btn("العربية", "ui|ar"), btn("English", "ui|en")]]
    return T("ar", "pick_ui"), InlineKeyboardMarkup(rows)


def screen_src(job):
    ui, key = job["ui"], job["id"]
    text = f"{T(ui, 'src_q')}\n\n{T(ui, 'src_note')}"
    rows = [
        [btn(T(ui, "src_auto"), f"src|{key}|auto")],
        [btn(T(ui, "src_manual"), f"src|{key}|pick")],
        *([[btn(T(ui, "quick", tgt=LANGS.get(job["tgt"], job["tgt"])), f"qk|{key}")]] if job.get("saved") else []),
        [btn(T(ui, "cancel"), f"x|{key}")],
    ]
    return text, InlineKeyboardMarkup(rows)


def screen_langs(job, codes, page, title_key, list_prefix, pick_prefix):
    ui, key = job["ui"], job["id"]
    chunk = codes[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    rows, row = [], []
    for c in chunk:
        row.append(btn(LANGS[c], f"{pick_prefix}|{key}|{c}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    nav = []
    if page > 0:
        nav.append(btn("◀", f"{list_prefix}|{key}|{page - 1}"))
    if (page + 1) * PAGE_SIZE < len(codes):
        nav.append(btn("▶", f"{list_prefix}|{key}|{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([btn(T(ui, "cancel"), f"x|{key}")])
    return T(ui, title_key), InlineKeyboardMarkup(rows)


def screen_src_list(job, page=0):
    return screen_langs(job, ordered(SOURCES, "en"), page, "pick_src", "sl", "ss")


def screen_tgt(job, page=0):
    return screen_langs(job, ordered(TARGETS, job["ui"]), page, "tgt_q", "tgl", "tg")


def screen_mode(job):
    ui, key = job["ui"], job["id"]
    rows = [
        [btn(T(ui, "mode_burn"), f"md|{key}|sub")],
        [btn(T(ui, "mode_txt"), f"md|{key}|txt")],
        [btn(T(ui, "cancel"), f"x|{key}")],
    ]
    return T(ui, "mode_q"), InlineKeyboardMarkup(rows)


def screen_settings(job):
    ui, key = job["ui"], job["id"]
    fonts = fonts_for(job["tgt"])
    font = fonts[job["font_i"] % len(fonts)]
    rows = [
        [btn(f"{T(ui, 'font')}: {font}", f"cy|{key}|font"), btn(T(ui, "font_preview"), f"fp|{key}")],
        [btn(f"{T(ui, 'size')}: {T(ui, SIZES[job['size_i']][0])}", f"cy|{key}|size")],
        [btn(f"{T(ui, 'style')}: {T(ui, 'style_' + STYLES[job['style_i']])}", f"cy|{key}|style"),
         btn(T(ui, "style_preview"), f"sp|{key}")],
        [btn(f"{T(ui, 'pos')}: {T(ui, POSITIONS[job['pos_i']][0])}", f"cy|{key}|pos")],
        [btn(f"🌐 {T(ui, 'bi')}: {T(ui, 'on' if job.get('bi') else 'off')}", f"cy|{key}|bi")],
        [btn(T(ui, "start"), f"go|{key}")],
        [btn(T(ui, "cancel"), f"x|{key}")],
    ]
    return T(ui, "settings"), InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------- handlers
def new_job(kind: str, data: str, ui: str, uid=None) -> dict:
    key = uuid.uuid4().hex[:10]
    pr = PREFS.get(str(uid), {}) if uid else {}
    JOBS[key] = {
        "id": key, "kind": kind, "data": data, "ui": ui,
        "src": None, "tgt": pr.get("tgt") if pr.get("tgt") in TARGETS else (ui if ui in TARGETS else "ar"),
        "mode": pr.get("mode", "sub"), "uid": uid,
        "font_i": int(pr.get("font_i", 0)), "size_i": min(int(pr.get("size_i", 1)), len(SIZES) - 1),
        "style_i": min(int(pr.get("style_i", 0)), len(STYLES) - 1), "pos_i": min(int(pr.get("pos_i", 0)), len(POSITIONS) - 1),
        "bi": bool(pr.get("bi", False)), "saved": bool(pr.get("tgt")),
    }
    while len(JOBS) > 300:
        JOBS.popitem(last=False)
    return JOBS[key]


async def safe_edit(q, text, markup=None):
    try:
        await q.edit_message_text(text, reply_markup=markup)
    except TelegramError as e:  # e.g. "message is not modified"
        log.debug("edit skipped: %s", e)


def get_ui(context, user):
    ui = context.user_data.get("ui")
    if not ui and user:
        ui = PREFS.get(str(user.id), {}).get("ui")
        if ui in STR:
            context.user_data["ui"] = ui
        else:
            ui = None
    return ui


async def cmd_glossary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg, user = update.effective_message, update.effective_user
    ui = get_ui(context, user) or "ar"
    pr = uprefs(user.id)
    gl = pr.setdefault("gl", {})
    body = (msg.text or "").split(None, 1)
    arg = body[1].strip() if len(body) > 1 else ""
    if arg.lower() == "clear":
        gl.clear()
        save_prefs()
        await msg.reply_text(T(ui, "gl_cleared"))
        return
    if arg.lower().startswith("del "):
        gl.pop(arg[4:].strip(), None)
        save_prefs()
        await msg.reply_text(T(ui, "gl_deleted"))
        return
    if arg:
        new = parse_glossary(arg)
        for a, b in new.items():
            if len(gl) < 60 or a in gl:
                gl[a] = b
        save_prefs()
        await msg.reply_text(f"{T(ui, 'gl_added')} ({len(new)})")
        return
    listing = "\n".join(f"• {a} ⇢ {b}" if a != b else f"• {a}" for a, b in gl.items()) or T(ui, "gl_none")
    await msg.reply_text(f"{T(ui, 'gl_help')}\n\n{T(ui, 'gl_list')}:\n{listing}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    ui = get_ui(context, update.effective_user)
    await send_banner(msg)
    if not ui:
        text, markup = screen_ui_picker()
        await msg.reply_text(text, reply_markup=markup)
    else:
        await msg.reply_text(T(ui, "welcome"))


async def cmd_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text, markup = screen_ui_picker()
    await update.effective_message.reply_text(text, reply_markup=markup)


async def on_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    kind = data = None
    if msg.video:
        kind, data = "file", msg.video.file_id
    elif msg.document and (msg.document.mime_type or "").startswith("video/"):
        kind, data = "file", msg.document.file_id
    elif msg.text:
        m = URL_RE.search(msg.text)
        if m:
            kind, data = "url", m.group(0)

    ui = get_ui(context, update.effective_user)
    if not kind:
        await msg.reply_text(T(ui or "ar", "send_media"))
        return
    if not ui:  # ask for the interface language first, keep the media for afterwards
        context.user_data["pending"] = (kind, data)
        text, markup = screen_ui_picker()
        await msg.reply_text(text, reply_markup=markup)
        return

    job = new_job(kind, data, ui, update.effective_user.id)
    text, markup = screen_src(job)
    await msg.reply_text(text, reply_markup=markup)


async def begin(context, q, job):
    JOBS.pop(job["id"], None)
    if job.get("uid"):
        uprefs(job["uid"]).update({k: job[k] for k in ("tgt", "mode", "font_i", "size_i", "style_i", "pos_i", "bi")})
        save_prefs()
    await safe_edit(q, T(job["ui"], "queued"))
    context.application.create_task(process(context.bot, q.message.chat_id, job, q.message))


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except TelegramError:
        pass
    parts = (q.data or "").split("|")
    act = parts[0]

    if act == "ui":
        ui = parts[1] if len(parts) > 1 and parts[1] in STR else "ar"
        context.user_data["ui"] = ui
        uprefs(q.from_user.id)["ui"] = ui
        save_prefs()
        pending = context.user_data.pop("pending", None)
        if pending:
            job = new_job(pending[0], pending[1], ui, q.from_user.id)
            text, markup = screen_src(job)
            await safe_edit(q, text, markup)
        else:
            await safe_edit(q, T(ui, "welcome"))
        return

    key = parts[1] if len(parts) > 1 else ""
    job = JOBS.get(key)
    if not job:
        await safe_edit(q, T(context.user_data.get("ui") or "ar", "expired"))
        return
    ui = job["ui"]

    if act == "x":
        JOBS.pop(key, None)
        await safe_edit(q, T(ui, "cancelled"))
    elif act == "src":
        if parts[2] == "auto":
            job["src"] = None
            text, markup = screen_tgt(job)
        else:
            text, markup = screen_src_list(job)
        await safe_edit(q, text, markup)
    elif act == "sl":
        text, markup = screen_src_list(job, int(parts[2]))
        await safe_edit(q, text, markup)
    elif act == "ss":
        job["src"] = parts[2] if parts[2] in LANGS else None
        text, markup = screen_tgt(job)
        await safe_edit(q, text, markup)
    elif act == "tgl":
        text, markup = screen_tgt(job, int(parts[2]))
        await safe_edit(q, text, markup)
    elif act == "tg":
        if parts[2] in TARGETS:
            job["tgt"] = parts[2]
            job["font_i"] = 0
        text, markup = screen_mode(job)
        await safe_edit(q, text, markup)
    elif act == "md":
        job["mode"] = "txt" if parts[2] == "txt" else "sub"
        if job["mode"] == "txt":
            await begin(context, q, job)
        else:
            text, markup = screen_settings(job)
            await safe_edit(q, text, markup)
    elif act == "cy":
        field = parts[2]
        if field == "font":
            job["font_i"] = (job["font_i"] + 1) % len(fonts_for(job["tgt"]))
        elif field == "size":
            job["size_i"] = (job["size_i"] + 1) % len(SIZES)
        elif field == "style":
            job["style_i"] = (job["style_i"] + 1) % len(STYLES)
        elif field == "pos":
            job["pos_i"] = (job["pos_i"] + 1) % len(POSITIONS)
        elif field == "bi":
            job["bi"] = not job.get("bi")
        text, markup = screen_settings(job)
        await safe_edit(q, text, markup)
    elif act == "fp":
        group = "ar" if job["tgt"] in ARABIC_SCRIPT else "lat"
        png = await font_preview(group)
        if png:
            await context.bot.send_photo(q.message.chat_id, photo=png, caption=T(ui, "font_preview_cap"))
    elif act == "sp":
        group = "ar" if job["tgt"] in ARABIC_SCRIPT else "lat"
        fonts = fonts_for(job["tgt"])
        png = await style_preview(group, fonts[job["font_i"] % len(fonts)], ui)
        if png:
            await context.bot.send_photo(q.message.chat_id, photo=png, caption=T(ui, "style_preview_cap"))
    elif act == "go":
        await begin(context, q, job)
    elif act == "qk":
        job["src"] = None
        await begin(context, q, job)


_STYLE_PREVIEW_CACHE: dict = {}


def _ass_style_line(name: str, font: str, size: int, defs: str) -> str:
    d = dict(kv.split("=", 1) for kv in defs.split(","))
    return (f"Style: {name},{font},{size},{d['PrimaryColour']},&H000000FF,{d['OutlineColour']},"
            f"{d.get('BackColour', '&H00000000')},1,0,0,0,100,100,0,0,{d['BorderStyle']},{d['Outline']},"
            f"{d['Shadow']},5,0,0,0,1")


async def style_preview(group: str, font: str, ui: str):
    """PNG with every subtitle style over a video-like background (cached)."""
    key = (group, font, ui)
    if key in _STYLE_PREVIEW_CACHE:
        return _STYLE_PREVIEW_CACHE[key]
    sample = "السلام عليكم" if group == "ar" else "Hello world"
    cols, cw, ch = 2, 360, 96
    rows = (len(STYLES) + cols - 1) // cols
    width, height = cols * cw, rows * ch
    size = round(34 * font_scale(font, group))
    label_font = "Noto Naskh Arabic" if ui == "ar" else "Noto Sans"
    styles, events = [], []
    for i, key_s in enumerate(STYLES):
        cx, cy = (i % cols) * cw, (i // cols) * ch
        styles.append(_ass_style_line(f"S{i}", font, size, STYLE_DEFS[key_s]))
        events.append(f"Dialogue: 1,0:00:00.00,0:00:05.00,S{i},,0,0,0,,{{\\an5\\pos({cx + cw // 2},{cy + ch // 2 + 8})}}{sample}")
        events.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,Lbl,,0,0,0,,{{\\an7\\pos({cx + 8},{cy + 4})\\fn{label_font}\\fs15}}{T(ui, 'style_' + key_s)}")
    ass = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Lbl,Noto Sans,15,&H00E0E0E0,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n"
        + "\n".join(styles)
        + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(events) + "\n"
    )
    workdir = Path(tempfile.mkdtemp(prefix="stylprev_"))
    try:
        (workdir / "p.ass").write_text(ass, encoding="utf-8")
        for bg in (f"gradients=s={width}x{height}:c0=0x2b5876:c1=0xc9a27a:x0=0:y0=0:x1={width}:y1={height}:d=1",
                   f"color=c=0x5a6b7b:s={width}x{height}:d=1"):
            rc, _, err = await run_capture(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", bg,
                                            "-vf", "ass=p.ass", "-frames:v", "1", "p.png"], cwd=str(workdir))
            if rc == 0 and (workdir / "p.png").exists():
                _STYLE_PREVIEW_CACHE[key] = (workdir / "p.png").read_bytes()
                return _STYLE_PREVIEW_CACHE[key]
        log.warning("style preview failed: %s", err.decode(errors="ignore")[-200:])
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


_PREVIEW_CACHE: dict = {}


async def font_preview(group: str):
    """PNG showing every offered font written in itself (cached). Same libass/ffmpeg path as the burn."""
    if group in _PREVIEW_CACHE:
        return _PREVIEW_CACHE[group]
    fonts = list(FONTS_AR if group == "ar" else FONTS_LAT)
    sample = "السلام عليكم ورحمة الله ١٢٣" if group == "ar" else "Hello world 123 - Aa Bb Cc"
    row_h, width = 64, 720
    height = row_h * len(fonts) + 16
    events = []
    for i, name in enumerate(fonts):
        y = 8 + row_h * i + row_h // 2
        size = round(30 * 1.2 * font_scale(name, group))
        events.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,L,,0,0,0,,{{\\an4\\pos(16,{y})\\fnNoto Sans\\fs20\\1c&H00B0B0B0&}}{i + 1}. {name}")
        events.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,S,,0,0,0,,{{\\an6\\pos({width - 16},{y})\\fn{name}\\fs{size}}}{sample}")
    ass = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: L,Noto Sans,20,&H00B0B0B0,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,4,0,0,0,1\n"
        "Style: S,Noto Sans,30,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,6,0,0,0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(events) + "\n"
    )
    workdir = Path(tempfile.mkdtemp(prefix="fontprev_"))
    try:
        (workdir / "p.ass").write_text(ass, encoding="utf-8")
        rc, _, err = await run_capture(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=0x1E1E1E:s={width}x{height}:d=1",
             "-vf", "ass=p.ass", "-frames:v", "1", "p.png"], cwd=str(workdir))
        png = workdir / "p.png"
        if rc == 0 and png.exists():
            _PREVIEW_CACHE[group] = png.read_bytes()
            return _PREVIEW_CACHE[group]
        log.warning("font preview failed: %s", err.decode(errors="ignore")[-200:])
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def run_capture(args, cwd=None, timeout=60):
    """Run a command and return (returncode, stdout, stderr) without raising."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        return proc.returncode, out, err
    except asyncio.TimeoutError:
        return -1, b"", b"timeout"
    except Exception as e:  # e.g. command not found
        return -1, b"", str(e).encode()


FONT_TEST_AR = "مرحبا بالعالم - السلام عليكم"


async def cmd_fonttest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Renders a sample of every subtitle font exactly as the server's ffmpeg/libass sees it."""
    msg = update.effective_message
    await msg.reply_text("🔎 Checking the server fonts...")
    workdir = Path(tempfile.mkdtemp(prefix="fonttest_"))
    try:
        names = CANDIDATES_AR + CANDIDATES_LAT
        styles, events = [], []
        for i, name in enumerate(names):
            styles.append(
                f"Style: f{i},{name},26,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
                "1,0,0,0,100,100,0,0,1,2,0,8,10,10,10,1"
            )
            sample = FONT_TEST_AR if name in CANDIDATES_AR else "Hello world 123"
            events.append(
                f"Dialogue: 0,0:00:00.00,0:00:02.00,f{i},,0,0,{15 + i * 55},,{name}: {sample}"
            )
        ass = (
            "[Script Info]\nScriptType: v4.00+\nPlayResX: 640\nPlayResY: 420\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            + "\n".join(styles)
            + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            + "\n".join(events) + "\n"
        )
        (workdir / "test.ass").write_text(ass, encoding="utf-8")
        rc, _, err = await run_capture(
            ["ffmpeg", "-y", "-v", "verbose", "-f", "lavfi", "-i", "color=c=black:s=640x420:d=1",
             "-vf", "ass=test.ass", "-frames:v", "1", "test.png"],
            cwd=str(workdir),
        )
        _, fl, _ = await run_capture(["fc-list", ":lang=ar", "family"])
        fams = sorted({f.split(",")[0].strip() for f in fl.decode(errors="ignore").splitlines() if f.strip()})
        lines = []
        for l in err.decode(errors="ignore").splitlines():
            if ("fontselect" in l or "Glyph" in l) and l.strip()[:160] not in lines:
                lines.append(l.strip()[:160])
        info = (
            f"ffmpeg exit code: {rc}\n"
            f"Fonts with Arabic support (fc-list): {', '.join(fams[:25]) or 'NONE'}\n"
            f"Offered in the bot: arabic={FONTS_AR} latin={FONTS_LAT}\n\n"
            "libass font selection:\n" + ("\n".join(lines[:12]) or "(no lines)")
        )
        # real-pipeline test: tiny portrait frame, SRT + subtitles filter, left = plain, right = supersampled
        (workdir / "t.srt").write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n{\\an5}السلام عليكم\nالاتصالات والمعلومات\n", encoding="utf-8")
        pngs = []
        for fi, fname in enumerate(CANDIDATES_AR):
            for ss in (1, 2):
                st = (f"FontName={fname},Bold=1,FontSize=14,{STYLE_DEFS['classic']},Alignment=2,MarginV=0,MarginL=20,MarginR=20")
                out_png = f"p{fi}_{ss}.png"
                if ss == 1:
                    vf = f"subtitles=t.srt:force_style='{st}'"
                else:
                    vf = f"scale=512:908,subtitles=t.srt:force_style='{st}',scale=256:454:flags=area"
                await run_capture(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                                   "color=c=0x303030:s=256x454:d=2", "-vf", vf, "-frames:v", "1", out_png],
                                  cwd=str(workdir))
                if (workdir / out_png).exists():
                    pngs.append(out_png)
        if len(pngs) == 2 * len(CANDIDATES_AR):
            args = ["ffmpeg", "-y", "-v", "error"]
            for n in pngs:
                args += ["-i", n]
            rows = len(pngs) // 2
            fc = "".join(f"[{2*i}][{2*i+1}]hstack[r{i}];" for i in range(rows))
            fc += "".join(f"[r{i}]" for i in range(rows)) + f"vstack=inputs={rows}"
            await run_capture(args + ["-filter_complex", fc, "pipeline.png"], cwd=str(workdir))
            if (workdir / "pipeline.png").exists():
                with open(workdir / "pipeline.png", "rb") as f:
                    await msg.reply_photo(f, caption="Real-pipeline test: left = normal, right = supersampled (rows: " + ", ".join(CANDIDATES_AR) + ")")
        png = workdir / "test.png"
        if png.exists():
            with open(png, "rb") as f:
                await msg.reply_photo(f, caption="Each line should show the font name + a sample")
        else:
            info += "\n\nno image produced:\n" + err.decode(errors="ignore")[-600:]
        await msg.reply_text(info[:3500])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def on_error(update, context: ContextTypes.DEFAULT_TYPE):
    log.error("handler error", exc_info=context.error)
    err = context.error
    await notify_admin(context.bot, f"⚠️ خطأ في البوت\n{type(err).__name__}: {str(err)[:400]}", key=f"h-{type(err).__name__}")


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(f"ID: {update.effective_user.id}")


async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        return
    up = int(time.time() - START_TIME)
    bad = [m for m, t in _llm_bad.items() if t > time.time()]
    await update.effective_message.reply_text(
        f"✅ شغّال منذ {up // 3600}س {up % 3600 // 60}د\n"
        f"الذاكرة الآن: {mem_now()} · الذروة: {mem_peak()}\n"
        f"الطلبات في الطابور/المعالجة: {len(PENDING)}\n"
        f"Groq: {'مفعّل' if GROQ_API_KEY else 'غير مفعّل'} · ASR={GROQ_ASR_MODEL}\n"
        f"نماذج الترجمة المعطّلة مؤقتًا: {', '.join(bad) or 'لا يوجد'}\n"
        f"الخطوط: {', '.join(FONTS_AR)}"
    )


async def filter_fonts():
    """Keep only fonts that are really installed, so we never offer one that renders as boxes."""
    rc, out, _ = await run_capture(["fc-list", ":", "family"])
    if rc != 0 or not out.strip():
        log.warning("fc-list failed; keeping all font options")
        return
    have = set()
    for line in out.decode(errors="ignore").splitlines():
        for fam in line.split(","):
            have.add(fam.strip().lower())
    for lst in (FONTS_AR, FONTS_LAT):
        keep = [f for f in lst if f.lower() in have]
        if keep:
            lst[:] = keep
        else:
            log.warning("none of %s found by fontconfig; leaving the list unchanged", lst)
    log.info("fonts offered: arabic=%s latin=%s", FONTS_AR, FONTS_LAT)


async def post_init(app: Application):
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            log.critical("%s not found in PATH - deploy with the Dockerfile!", tool)
    await filter_fonts()
    if GROQ_API_KEY:
        log.info("Groq enabled: ASR=%s LLM=%s (local Whisper loads only as a fallback)", GROQ_ASR_MODEL, GROQ_LLM_MODEL)
    else:
        await asyncio.to_thread(get_model)  # warm up so the first request is fast
    log.info("startup complete, memory in use: %s", mem_now())
    await notify_admin(app.bot, f"🟢 البوت اشتغل (تشغيل جديد)\nالذاكرة: {mem_now()}", key="startup")


# -------------------------------------------------------------------- main
def main():
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(post_init)
        .concurrent_updates(True)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(300)  # video upload
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("language", cmd_language))
    app.add_handler(CommandHandler("fonttest", cmd_fonttest))
    app.add_handler(CommandHandler("glossary", cmd_glossary))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("diag", cmd_diag))
    app.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE
            & ((filters.TEXT & ~filters.COMMAND) | filters.VIDEO | filters.Document.VIDEO),
            on_input,
        )
    )
    app.add_handler(
        CallbackQueryHandler(on_button, pattern=r"^(ui|src|sl|ss|tgl|tg|md|cy|fp|sp|go|qk|x)(\||$)")
    )
    app.add_error_handler(on_error)

    if WEBHOOK_URL:
        install_health_route()
        load_prefs()
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TOKEN,
            webhook_url=f"{WEBHOOK_URL}/{TOKEN}",
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES,
        )
    else:  # local testing
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
