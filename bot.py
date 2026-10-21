"""SubVid Ai - Telegram bot: burn subtitles into a video or extract them as text.

Flow: pick UI language -> send video/link -> source language (auto/manual)
      -> target language -> output type -> (style settings) -> process.
Pipeline: download -> audio (ffmpeg) -> transcribe (faster-whisper)
          -> translate (Google API / gtx / MyMemory) -> clean + wrap -> burn or send text.
"""
import asyncio
import base64
import collections
import gc
import math
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
from datetime import datetime, timedelta, timezone
from collections import OrderedDict, deque
from pathlib import Path

import httpx
import yt_dlp
from faster_whisper import WhisperModel
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
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
BASE_DIR = Path(__file__).resolve().parent


def _find_tgapi():
    """telegram-bot-api binary: built into the image, or built later by /buildlocal into /app/bin (survives restarts)."""
    for c in (BASE_DIR / "bin" / "telegram-bot-api", shutil.which("telegram-bot-api")):
        if c and Path(c).exists():
            return str(c)
    return None


TGAPI_BIN = _find_tgapi()


def _load_local_api():
    """Own Telegram Bot API server (no 20MB/50MB limits). Active only if api_id/api_hash were saved AND the binary exists."""
    try:
        d = json.loads((BASE_DIR / "localapi.json").read_text(encoding="utf-8"))
        i, h = str(d["api_id"]).strip(), str(d["api_hash"]).strip()
        if i.isdigit() and re.fullmatch(r"[0-9a-fA-F]{32}", h) and TGAPI_BIN:
            return i, h
    except Exception:
        pass
    return None


LOCAL_API = _load_local_api()
LOCAL = LOCAL_API is not None
FILE_LIMIT_TXT = "2GB" if LOCAL else "20MB"
MAX_RES = int(os.environ.get("VIDEO_MAX_RES", "1080"))  # max size of the SMALLER video side (480 = 480x854 portrait)
USE_VAD = os.environ.get("USE_VAD", "1") != "0"  # "0" skips the VAD model (saves RAM, may add noise text)
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "2"))
DEBUG_TEXT = os.environ.get("DEBUG_TEXT") == "1"
MAX_MINUTES = int(os.environ.get("CLIP_MAX_MINUTES", "60" if LOCAL else "20"))          # burn-in video (slow on the free CPU)
MAX_MINUTES_TXT = int(os.environ.get("MAX_MINUTES_TXT", "25"))  # text + SRT only (Groq accepts files up to 25MB)
COOKIES_FILE = os.environ.get("YTDLP_COOKIES")  # optional cookies.txt for yt-dlp
BURN_THREADS = os.environ.get("BURN_THREADS", "1")   # raise on a bigger server (e.g. 4)
BURN_PRESET = os.environ.get("BURN_PRESET", "ultrafast")  # ultrafast = fastest encode (slightly bigger file)
SS = max(1, min(3, int(os.environ.get("SUB_SUPERSAMPLE", "1"))))  # render subtitles at SSx size, then shrink (avoids glyph boxes at tiny sizes)
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0") or 0)   # your Telegram numeric id (send /id to the bot); receives error alerts
START_TIME = time.time()
def _secret(name: str):
    """Keys saved from Telegram with /setgroq live in secrets.json next to the code (overrides the environment)."""
    try:
        return json.loads((BASE_DIR / "secrets.json").read_text(encoding="utf-8")).get(name)
    except Exception:
        return None


GROQ_API_KEY = _secret("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")  # optional: far more accurate transcription + translation
GROQ_ASR_MODEL = os.environ.get("GROQ_ASR_MODEL", "whisper-large-v3")  # most accurate; turbo is used as fallback
GROQ_ASR_FALLBACK = "whisper-large-v3-turbo"
SUB_DELAY = float(_secret("SUB_DELAY") if _secret("SUB_DELAY") is not None else os.environ.get("SUB_DELAY", "0.06"))  # seconds subtitles are shifted later (text was appearing before the voice)
GROQ_PROMPT = os.environ.get("GROQ_PROMPT", "")  # optional: names/terms that appear in your clips
GROQ_LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b")  # llama-3.3-70b gave model_not_found on this account
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # optional official Google Cloud Translation key
TG_UPLOAD_LIMIT = (1900 if LOCAL else 49) * 1024 * 1024  # cloud Bot API upload limit is 50 MB, own server 2000 MB

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("subvid")


class _RingLog(logging.Handler):
    """Keeps the last log lines in memory so the admin can get them with /logs (no server access needed)."""
    def __init__(self):
        super().__init__()
        self.lines = collections.deque(maxlen=int(os.environ.get("LOG_LINES", "3000")))

    def emit(self, record):
        try:
            self.lines.append(self.format(record)[:600])
        except Exception:
            pass


RING = _RingLog()
RING.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.getLogger().addHandler(RING)

URL_RE = re.compile(r"https?://\S+")
JOB_SEM = asyncio.Semaphore(int(os.environ.get("JOB_CONCURRENCY", "1")))  # one heavy job at a time (protects RAM/CPU)
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
SIZES = [("size_s", 1.02), ("size_m", 1.44), ("size_l", 1.98)]   # +20% (was 0.85 / 1.2 / 1.65)
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
        "pick_ui": "🌐 اختر لغة الواجهة | Choose the interface language",
        "welcome": "✅ تم اختيار اللغة.\n\n🎬 ماذا يفعل SubVid Ai؟\n🎙 يستخرج الكلام من الفيديو بدقة (أي لغة)\n🌍 يترجمه إلى لغتك مع مراجعة تلقائية للأخطاء\n🎞 يدمج الترجمة على الفيديو كما في الأفلام، بالخط والحجم والنمط والمكان الذي تختاره\n🌐 ترجمة ثنائية اللغة (لغتين معًا) إذا حبيت\n📝 أو يعطيك النص مع ملف SRT لتعدّله ثم يعيد دمجه\n\n📥 ما المدعوم؟\n• رابط تيك توك\n• رابط فيديو مباشر\n• ملف فيديو من جوالك (حتى " + FILE_LIMIT_TXT + ")\n• يوتيوب غير مدعوم حاليًا، أرسل الفيديو كملف\n\n⏱ أقصى مدة للمقطع: " + str(MAX_MINUTES) + " دقائق\n\n▶️ ابدأ الآن: أرسل الرابط أو الفيديو.\n\n🔒 الخصوصية: تُحذف الفيديوهات من السيرفر بعد المعالجة (ويُحفظ الأخير ساعتين فقط إذا استخرجت نصه).\n⏳ إذا كان البوت نائمًا فقد يتأخر أول رد نحو دقيقة.\n\nلتغيير لغة الواجهة: /language",
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
        "start": "▶️ ابدأ", "back": "🔙 رجوع", "cancel": "إلغاء", "cancelled": "تم الإلغاء.",
        "expired": "انتهت صلاحية الطلب، أعد إرسال الفيديو.",
        "queued": "⏳ جاري التجهيز...",
        "queue": "⏳ ترتيبك في الطابور: {n} قبلك.",
        "proc_title": "⏳ جاري المعالجة...",
        "limit_day": "وصلت حدّك اليومي ({n} مقاطع). جرّب غدًا.", "busy_user": "عندك مقطع قيد المعالجة. انتظر حتى ينتهي ثم أرسل الثاني.",
        "not_allowed": "هذا البوت خاص حاليًا.\nمعرّفك: {id}\nأرسله لصاحب البوت ليضيفك.",
        "busy": "الخدمة مشغولة الآن (حدّ الاستخدام المجاني). جرّب بعد دقائق.",
        "retry": "🔄 إعادة المحاولة",
        "preview": "👁 معاينة مباشرة", "preview_all": "ℹ️ توضيح", "live_cap": "👁 معاينة مباشرة — تتحدث تلقائيًا مع كل تغيير في الخط أو الحجم أو النمط أو المكان أو الثنائية", "preview_cap": "الخطوط · الأحجام · الأنماط · المكان · الثنائية",
        "pv_fonts": "الخطوط", "pv_sizes": "الأحجام", "pv_styles": "الأنماط", "pv_pos": "المكان", "pv_bi": "ثنائية اللغة",
        "pv_bi_note": "الترجمة فوق\\Nواللغة الثانية تحتها بخط أصغر", "waking": "⏳ البوت كان في وضع الراحة وبدأ للتو، فالمعالجة الأولى قد تكون أبطأ قليلًا.",
        "tx_offer": "تبي تعدّل النص؟ استخرجه وصحّحه وأعد إرساله لأدمجه من جديد.", "tx_btn": "📝 استخراج النص",
        "tx_how": "✏️ عدّل النص ثم أرسله لي (كرسالة أو كملف .txt/.srt).\n• السطور التي تبدأ بـ | هي النص الأصلي (في الترجمة الثنائية).\n• لا تغيّر أرقام الأسطر ولا تحذف سطرًا أو تضف سطرًا، وسأبقي التوقيت كما هو.\n• أو أرسل ملف SRT كاملًا بتوقيتك الخاص.\n• الملف محفوظ لساعتين.",
        "tx_expired": "انتهت صلاحية هذا الفيديو (يُحفظ ساعتين). أرسل الفيديو من جديد.",
        "tx_mismatch": "عدد المقاطع في النص ({n}) لا يطابق الأصل ({m}). أبقِ نفس الترقيم، أو أرسل SRT بتوقيتك.",
        "caption_edit": "✅ تم تطبيق تعديلاتك.",
        "bi": "ثنائية اللغة", "bi_src": "اللغة الأصلية للفيديو", "bi_off": "إيقاف", "bi_back": "↩ رجوع",
        "bi_pick": "🌐 اختر اللغة الثانية التي تظهر تحت الترجمة بخط أصغر.\n«اللغة الأصلية» تعرض ما قيل في الفيديو بعد تصحيحه.", "on": "مفعّلة", "off": "متوقفة",
        "quick": "🔁 بإعداداتي السابقة (تلقائي ➜ {tgt})", "fast": "⚡ بدء سريع (كل الإعدادات افتراضية)",
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
        "yt_block": "⚠️ يوتيوب يمنع التحميل من سيرفر البوت حاليًا.\nأرسل الفيديو كملف مباشرة، أو استخدم رابط تيك توك أو رابطًا مباشرًا للفيديو.",
        "dl_empty": "لم يتم العثور على فيديو في هذا الرابط.",
        "file_big": "تيليجرام يمنع البوت من تنزيل ملفات أكبر من 20MB. ارفع الفيديو على أي موقع وأرسل لي الرابط بدلاً من الملف.",
        "translate_fail": "تعذرت الترجمة حالياً (خدمة الترجمة لا تستجيب). حاول بعد قليل.",
        "timeout": "استغرقت المعالجة وقتاً أطول من المسموح، جرّب مقطعاً أقصر.",
        "generic": "حصل خطأ غير متوقع أثناء المعالجة. حاول مرة ثانية.",
    },
    "en": {
        "pick_ui": "🌐 اختر لغة الواجهة | Choose the interface language",
        "welcome": "✅ Language set.\n\n🎬 What does SubVid Ai do?\n🎙 Transcribes the speech in your video accurately (any language)\n🌍 Translates it into your language with an automatic error review\n🎞 Burns the subtitles onto the video just like in movies, with the font, size, style and position you choose\n🌐 Bilingual subtitles (two languages together) if you like\n📝 Or gives you the text plus an SRT file to edit and burn again\n\n📥 What is supported?\n• TikTok links\n• Direct video links\n• A video file from your phone (up to " + FILE_LIMIT_TXT + ")\n• YouTube is not supported right now; send the video as a file\n\n⏱ Maximum clip length: " + str(MAX_MINUTES) + " minutes\n\n▶️ Start now: send the link or the video.\n\n🔒 Privacy: videos are deleted from the server after processing (the latest one is kept for 2 hours only if you export its text).\n⏳ If the bot was asleep, the first reply can take up to a minute.\n\nChange interface language: /language",
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
        "start": "▶️ Start", "back": "🔙 Back", "cancel": "Cancel", "cancelled": "Cancelled.",
        "expired": "This request expired, please send the video again.",
        "queued": "⏳ Preparing...",
        "queue": "⏳ Your place in the queue: {n} ahead of you.",
        "proc_title": "⏳ Processing...",
        "limit_day": "You've reached your daily limit ({n} clips). Try again tomorrow.", "busy_user": "You already have a clip being processed. Wait until it finishes.",
        "not_allowed": "This bot is private for now.\nYour ID: {id}\nSend it to the bot owner to get access.",
        "busy": "The service is busy right now (free usage limit). Try again in a few minutes.",
        "retry": "🔄 Try again",
        "preview": "👁 Live preview", "preview_all": "ℹ️ Guide", "live_cap": "👁 Live preview — updates automatically whenever you change font, size, style, position or bilingual", "preview_cap": "Fonts · Sizes · Styles · Position · Bilingual",
        "pv_fonts": "Fonts", "pv_sizes": "Sizes", "pv_styles": "Styles", "pv_pos": "Position", "pv_bi": "Bilingual",
        "pv_bi_note": "Translation on top\\Nsecond language below, smaller", "waking": "⏳ The bot was resting and just woke up, so the first job may be a little slower.",
        "tx_offer": "Want to edit the text? Export it, fix it, and send it back to burn it again.", "tx_btn": "📝 Export text",
        "tx_how": "✏️ Edit the text and send it back (as a message or a .txt/.srt file).\n• Lines starting with | are the original text (bilingual mode).\n• Keep the numbering: don't add or delete entries, timing stays the same.\n• Or send a full SRT with your own timing.\n• The file is kept for 2 hours.",
        "tx_expired": "This video has expired (kept for 2 hours). Please send it again.",
        "tx_mismatch": "The text has {n} entries but the original has {m}. Keep the same numbering, or send an SRT with timing.",
        "caption_edit": "✅ Your edits were applied.",
        "bi": "Bilingual", "bi_src": "Original language", "bi_off": "Off", "bi_back": "↩ Back",
        "bi_pick": "🌐 Choose the second language shown under the translation in a smaller font.\n\"Original language\" shows what was said in the video, corrected.", "on": "On", "off": "Off",
        "quick": "🔁 With my last settings (auto ➜ {tgt})", "fast": "⚡ Quick start (all default settings)",
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
        "yt_block": "⚠️ YouTube is blocking downloads from the bot's server right now.\nSend the video as a file, or use a TikTok link or a direct video link.",
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


_HALLU_CONTAINS = [
    "اشتركوا في القناه", "اشترك في القناه", "اشتركو في القناه", "اشتركوا بالقناه", "اشترك بالقناه", "لا تنسوا الاشتراك",
    "لا تنسى الاشتراك", "لا تنسو الاشتراك", "شكرا للمشاهده", "شكرا لمشاهدتكم", "شكرا على المشاهده", "شكرا لكم على المشاهده",
    "ترجمه نانسي", "ترجمه هدى", "اشتركوا بالقناه", "thanks for watching", "thank you for watching", "please subscribe",
    "like and subscribe", "subscribe to the channel", "amara org", "subtitles by", "subtitled by", "subtitles made by",
]
_HALLU_EXACT = {"ترجمه", "الترجمه", "اشتركوا", "اشترك", "subscribe", "bye", "thank you", "thanks", "you", "شكرا", "شكرا لكم"}


def _norm_h(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower().translate(_AR_MARKS_MAP)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def is_hallucination(text: str, edge: bool, no_speech: float, logprob: float, comp: float) -> bool:
    """Whisper invents stock phrases ("subscribe to the channel") on music/noise, mostly at the start or end of a clip."""
    if comp and comp > 2.4:                      # repetition loop
        return True
    if no_speech > 0.6 and logprob < -1.0:       # probably no speech and the model is unsure
        return True
    n = _norm_h(text)
    if not n:
        return True
    if n in _HALLU_EXACT or (len(n) <= 70 and any(ph in n for ph in _HALLU_CONTAINS)):
        return edge or logprob < -0.45 or no_speech > 0.25
    return False


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
            data=data, files={"file": ("audio.ogg", f, "audio/ogg")}, timeout=120,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def transcribe_groq(audio_path: str, language=None, vocab: str = "", words_out=None):
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
    words = [(float(w["start"]), float(w["end"]), str(w.get("word", ""))) for w in d.get("words", []) if "start" in w]
    out = []
    raw = [x for x in d.get("segments", []) if x.get("text", "").strip() and float(x.get("no_speech_prob", 0)) < 0.8]
    for i, x in enumerate(raw):
        text = x.get("text", "").strip()
        if is_hallucination(text, i in (0, len(raw) - 1), float(x.get("no_speech_prob", 0)),
                            float(x.get("avg_logprob", 0)), float(x.get("compression_ratio", 0))):
            log.info("dropped a likely hallucination: %r", text[:60])
            continue
        a, b = float(x["start"]), float(x["end"])
        inside = [w for w in words if a - 0.05 <= (w[0] + w[1]) / 2 <= b + 0.05]
        if inside:  # word timing is tighter than the segment edges -> subtitle matches the voice
            a, b = inside[0][0], inside[-1][1]
        out.append((a + SUB_DELAY, b + SUB_DELAY, text))
        if words_out is not None and inside:
            words_out[(a + SUB_DELAY, b + SUB_DELAY)] = [(u + SUB_DELAY, v + SUB_DELAY, t) for u, v, t in inside]
    lang = language or LANG_NAME_TO_CODE.get(str(d.get("language", "")).lower(), str(d.get("language", ""))[:2])
    if DEBUG_TEXT:
        log.info("second opinion (%s): %s", other, alt[:400])
    return out, lang, alt


def transcribe(audio_path: str, language=None, vocab: str = "", words_out=None):
    """language=None -> auto-detect; otherwise skip detection (faster)."""
    if GROQ_API_KEY:
        try:
            return transcribe_groq(audio_path, language, vocab, words_out)
        except Exception as e:
            if "429" in str(e):   # free-tier limit: local Whisper on this tiny CPU would take forever
                raise UserError("busy")
            log.warning("Groq transcription failed, using local Whisper: %s", str(e)[:150])
    segments, info = get_model().transcribe(
        audio_path,
        beam_size=1,
        vad_filter=USE_VAD,
        condition_on_previous_text=False,
        language=language,
        word_timestamps=True,
    )
    out = []
    segs = [x for x in segments if x.text.strip()]
    for i, s in enumerate(segs):
        if is_hallucination(s.text, i in (0, len(segs) - 1), float(s.no_speech_prob or 0), float(s.avg_logprob or 0),
                            float(s.compression_ratio or 0)):
            log.info("dropped a likely hallucination: %r", s.text.strip()[:60])
            continue
        out.append((s.start, s.end, s.text.strip()))
        if words_out is not None and getattr(s, "words", None):
            words_out[(s.start, s.end)] = [(w.start, w.end, w.word) for w in s.words]
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


_LLM_MODELS = [m for m in dict.fromkeys([GROQ_LLM_MODEL, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]) if m]
_llm_bad: dict = {}   # model -> time until which it is skipped


def _rate_wait(text: str) -> float:
    """Seconds Groq asks us to wait ("Please try again in 7.2s" / "in 350ms" / "in 1m3s"), 0 if unknown."""
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)(ms|s)", text or "")
    if not m:
        return 0.0
    mins = float(m.group(1) or 0)
    val = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
    return mins * 60 + val


def _groq_chat(body: dict, timeout: int) -> dict:
    """POST to Groq; on a rate limit that clears within ~25 s, wait and retry once instead of giving up."""
    for attempt in range(2):
        r = httpx.post("https://api.groq.com/openai/v1/chat/completions",
                       headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json=body, timeout=timeout)
        if r.status_code == 429 and attempt == 0:
            w = _rate_wait(r.text)
            if 0 < w <= 25:
                log.info("Groq rate limit, waiting %.1fs", w)
                time.sleep(w + 0.5)
                continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()


def _llm_batch(texts, src_name, tgt_name, fix_only, alt="", gloss=None):
    task = (
        f"These are consecutive subtitle lines from a video, auto-transcribed from speech ({src_name}). "
        "The transcription may contain mistakes (wrong or misheard words, names, missing letters). "
        "If a word looks garbled, replace it with the most likely intended word given the context. "
    )
    task += ("NEVER delete or skip words or sentences: every spoken word must stay in the output. " if fix_only else
             "Do not drop any information; only pure fillers (so, well, now, you know, um) may be left out. ")
    if alt:
        task += ("A second speech-recognition model transcribed the same audio as: \"" + alt[:1500] + "\". "
                 "Where the lines miss a word or phrase that the second transcript has and it fits the context, "
                 "put it back into the right line. ")
    if gloss:
        task += ("Corrections already found for this transcript - always use exactly these terms/spellings (format: heard => correct): "
                 + "; ".join(f"{a} => {b}" if a != b else a for a, b in list(gloss.items())[:40]) + ". ")
    if fix_only:
        task += "Fix obvious transcription mistakes using the context. Keep the same language and meaning."
    else:
        task += (f"First fix obvious mistakes using the context, then translate each line into {tgt_name} the way a professional "
                 "subtitler would: translate the MEANING, not word by word, and re-order phrases as the target language "
                 "requires; use correct grammar and agreement (in Arabic: verb-subject agreement, correct prepositions, "
                 "natural word order) and the standard term for technical, military and political vocabulary. Write names of "
                 "people, organizations and media outlets in the target script when they have a common form (for example "
                 "Arab News -> عرب نيوز), but keep acronyms such as ISR or NASA as they are. The lines are consecutive pieces "
                 "of one speech; each line is a whole sentence or phrase.")
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
            out = json.loads(_groq_chat(body, 60)["choices"][0]["message"]["content"])["t"]
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


def _llm_json(system: str, payload: dict, timeout: int = 60) -> dict:
    """One JSON-mode call to the first available Groq model."""
    last = None
    for model in _LLM_MODELS:
        if _llm_bad.get(model, 0) > time.time():
            continue
        body = {"model": model, "temperature": 0.1, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
        try:
            return json.loads(_groq_chat(body, timeout)["choices"][0]["message"]["content"])
        except Exception as e:
            last = e
            log.warning("Groq LLM %s failed: %s", model, str(e)[:200])
            if "HTTP 429" in str(e):
                global LLM_LIMITED_AT
                LLM_LIMITED_AT = time.time()
            if "HTTP 404" in str(e) or "HTTP 403" in str(e):
                _llm_bad[model] = time.time() + 3600
    raise last or RuntimeError("no LLM model available")


LLM_LIMITED_AT = 0.0
AUTO_REVIEW = os.environ.get("AUTO_REVIEW", "1") != "0"


def auto_fixes(segments, alt: str, src_name: str) -> dict:
    """Automatic proofreading of the raw transcript: finds misheard / garbled words (using the whole context and the
    second speech model's version) and returns {wrong text: right text}. Replaces the old manual glossary."""
    if not GROQ_API_KEY or not AUTO_REVIEW:
        return {}
    texts = [" ".join(t.split()) for _, _, t in segments][:400]
    system = (
        f"You proofread an automatic speech transcript ({src_name}) split into numbered lines. Speech recognition often "
        "mishears words (similar sounds), garbles names and terms, or drops words. Use the whole context, the topic and the "
        "second transcript of the same audio (if given) to find the wrong words. Return JSON "
        '{"fixes": [{"wrong": "...", "right": "..."}]} with at most 25 items. "wrong" MUST be copied exactly, character '
        "for character, from the lines (a word or a short phrase, 1-4 words), and \"right\" its most likely correct form in "
        "the same language. Only include fixes you are confident about; do not rewrite style or grammar. If nothing needs "
        'fixing return {"fixes": []}.')
    try:
        d = _llm_json(system, {"lines": texts, "second_transcript": alt[:3000]})
    except Exception as e:
        log.warning("auto proofreading skipped: %s", str(e)[:150])
        return {}
    joined = "\n".join(texts)
    out = {}
    for f in d.get("fixes", [])[:25]:
        try:
            w, r = str(f["wrong"]).strip(), str(f["right"]).strip()
        except Exception:
            continue
        if w and r and w != r and w in joined and len(w) <= 60 and len(r) <= 80:
            out[w] = r
    log.info("auto proofreading: %d fixes", len(out))
    if DEBUG_TEXT:
        log.info("fixes: %s", out)
    return out


def review_translation(sources, translated, alt: str, src_name: str, tgt_name: str):
    """Second pass: a strict reviewer compares each source line with its translation and corrects mistakes.
    Keeps the same number of lines; any problem -> the original translation is kept."""
    if not GROQ_API_KEY or not AUTO_REVIEW or not translated or len(translated) > 160:
        return translated
    system = (
        f"You are a senior subtitle editor and a native {tgt_name} speaker. For each numbered item you get the source line "
        f"({src_name}) and its translation into {tgt_name}. Fix: mistranslations, grammar and agreement mistakes, "
        "word-for-word literal phrasing (rewrite it as a native speaker would naturally say it), information missing from the "
        "translation (put it back), wrong names, and Latin-script names that should be written in the target script (keep "
        "acronyms). Keep each item the same sentence: do NOT merge, split or reorder items; the length may change by up to 30%. "
        'If a translation is already natural and correct, return it unchanged. Return JSON {"t": [...]} with EXACTLY the same number of items.')
    out = []
    for i in range(0, len(translated), 30):
        src_b, tr_b = sources[i:i + 30], translated[i:i + 30]
        try:
            d = _llm_json(system, {"items": [{"source": a, "translation": b} for a, b in zip(src_b, tr_b)],
                                   "second_transcript": alt[:1500] if len(translated) <= 30 else ""})
            t = d.get("t")
            if not isinstance(t, list) or len(t) != len(tr_b):
                raise ValueError("line count mismatch")
            out += [str(x).strip() or o for x, o in zip(t, tr_b)]
        except Exception as e:
            log.warning("review skipped for a batch: %s", str(e)[:150])
            out += tr_b
    return out


def translate_segments_llm(segments, src: str, tgt: str, alt: str = "", gloss=None):
    """Context-aware translation + correction via Groq. Raises on failure."""
    names = {c: n for n, c in LANG_NAME_TO_CODE.items()}
    src_name, tgt_name = names.get(src, src), names.get(tgt, tgt)
    texts = [" ".join(t.split()) for _, _, t in segments]
    def run(batch):
        """A batch that fails (wrong line count, bad JSON) is split in two and retried: small batches rarely fail."""
        try:
            return _llm_batch(batch, src_name, tgt_name, fix_only=(src == tgt), alt=alt if len(texts) <= 30 else "", gloss=gloss)
        except Exception:
            if len(batch) <= 3:
                raise
            time.sleep(1)
            mid = len(batch) // 2
            return run(batch[:mid]) + run(batch[mid:])

    result = []
    for i in range(0, len(texts), 30):
        result += run(texts[i:i + 30])
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


_END_BAD = {  # a line/cue should not end with these (they belong to the next word)
    "في", "على", "من", "إلى", "الى", "عن", "مع", "بين", "حتى", "عبر", "خلال", "نحو", "ذلك", "هذا", "هذه", "أن", "ان", "إن",
    "لم", "لن", "لا", "ما", "قد", "كل", "بعد", "قبل", "عند", "ضد", "منذ", "حول", "لدى", "أو", "او", "ثم", "و", "ب", "ل",
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "and", "or", "but", "that", "is", "are", "was", "by",
    "from", "as", "this", "these", "those", "his", "her", "their", "our", "my", "your", "its", "not", "de", "la", "el", "en",
}
_TITLES = {  # keep a title together with the name that follows
    "اللواء", "الركن", "الدكتور", "الشيخ", "الأمير", "الملك", "الرئيس", "وزير", "المتحدث", "الجنرال", "العميد", "المقدم",
    "الفريق", "السيد", "الأستاذ", "المهندس", "الكابتن", "mr", "mrs", "dr", "general", "president", "king", "prince",
}
_CONJ_START = {"ثم", "لكن", "حيث", "مما", "الذي", "التي", "الذين", "بينما", "أو", "إلا", "كما", "لأن", "عندما", "بعدما",
               "and", "but", "which", "because", "while", "when", "then", "so"}
_PUNCT_END = ".,;:!?،؛؟…"


def break_cost(prev: str, nxt: str) -> float:
    """How bad it is to cut a subtitle between these two words (lower = better place to cut)."""
    c = 0.0
    if prev and prev[-1] in ".!?؟…。！？":
        c -= 3.5                                   # end of a sentence: the best place to cut
    elif prev and prev[-1] in _PUNCT_END:
        c -= 1.5
    if re.match(r"^[A-Za-z][A-Za-z.\-']*$", prev.strip(_PUNCT_END)) and re.match(r"^[A-Za-z]", nxt) and prev[-1:] not in _PUNCT_END:
        c += 3.0                                   # never cut inside a Latin name such as "Arab News"
    p = prev.strip(_PUNCT_END + "\"'()[]").lower()
    n = nxt.strip(_PUNCT_END + "\"'()[]").lower()
    if p in _END_BAD or p in _TITLES:
        c += 2.0
    if n in _CONJ_START or (len(n) > 3 and n[0] == "و" and "\u0600" <= n[0] <= "\u06ff"):
        c -= 0.8
    return c


def wrap2(line: str, max_chars: int) -> str:
    """Wrap into at most two balanced lines, cutting at a natural place (not after a preposition, not inside a title)."""
    if len(line) <= max_chars:
        return line
    words = line.split()
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        cost = abs(len(a) - len(b)) / max(1, len(line)) * 4 + break_cost(words[i - 1], words[i])
        if best is None or cost < best[0]:
            best = (cost, a, b)
    return f"{best[1]}\n{best[2]}" if best else line


def chunk_words(words, limit: int, start: float, end: float, sw):
    """Cuts a word list into cues of at most `limit` characters at natural places, with even sizes and no orphan words.
    Returns [(j, i)] index ranges. sw (real word times) lets it avoid cues that would be too short to read."""
    n = len(words)
    pref = [0]
    for w in words:
        pref.append(pref[-1] + len(w) + 1)

    def L(j, i):
        return pref[i] - pref[j] - 1

    total = L(0, n)
    if total <= limit:
        return [(0, n)]
    target = total / math.ceil(total / limit)
    nw = len(sw)

    def t_at(i):
        if i >= n:
            return end
        return sw[min(nw - 1, round(i * nw / n))][0] if nw >= 2 else start + (end - start) * i / n

    INF = float("inf")
    dp, prev = [INF] * (n + 1), [-1] * (n + 1)
    dp[0] = 0.0
    for i in range(1, n + 1):
        for j in range(i - 1, -1, -1):
            l = L(j, i)
            if l > limit and i - j > 1:
                break
            c = ((l - target) / limit) ** 2 * 8 + 1.0
            if i < n:
                c += break_cost(words[i - 1], words[i])
            if i - j == 1 and l < 0.4 * limit:
                c += 3.0
            d = t_at(i) - t_at(j)
            if d < 1.3:
                c += (1.3 - d) * 6                 # a cue shown for under ~1.3 s cannot be read
            if dp[j] + c < dp[i]:
                dp[i], prev[i] = dp[j] + c, j
    out, i = [], n
    while i > 0:
        out.append((prev[i], i))
        i = prev[i]
    return out[::-1]


MIN_CUE = 0.6   # seconds: a one-line cue shorter than this is hard to read, so the segment keeps two lines instead


def merge_sentences(segments, wmap):
    """Joins Whisper's segments (often cut in the middle of a sentence) into whole sentences, so each sentence is translated
    and shown as one unit. Returns (segments, wmap) with the same structure."""
    out, new_map, group = [], {}, []

    def flush():
        if not group:
            return
        a, b = group[0][0], group[-1][1]
        text = " ".join(t for _, _, t in group)
        ws = []
        for x, y, _ in group:
            for w in wmap.get((x, y)) or []:
                if not ws or w[0] > ws[-1][0] + 1e-6:      # neighbouring segments may share a boundary word
                    ws.append(w)
        out.append((a, b, text))
        if len(ws) >= 2:
            new_map[(a, b)] = ws
        group.clear()

    for seg in segments:
        group.append(seg)
        text = " ".join(t for _, _, t in group)
        done = text.rstrip().endswith((".", "!", "?", "؟", "…", "。", "！", "？", ".\"", "?\"", "!\""))
        if done or len(text) > 260 or group[-1][1] - group[0][0] > 14:
            flush()
    flush()
    return out, new_map


def _plan_cues(words, start, end, limit, sw):
    """Cuts one spoken segment into cues of at most `limit` characters. Returns [(t, duration, line, p0, p1)] where p0..p1 is
    the share of the text. With real word times (sw) the cuts fall where the speaker says the words; otherwise by text length."""
    ranges = chunk_words(words, limit, start, end, sw)
    chunks = [words[j:i] for j, i in ranges]
    total = sum(len(" ".join(c)) for c in chunks) or 1
    t, dur, cum, plan = start, end - start, 0, []
    for k, c in enumerate(chunks):
        line = " ".join(c)
        d = dur * len(line) / total
        if len(sw) >= 2 and len(chunks) > 1:
            nw = len(sw)
            i0 = min(nw - 1, round(nw * cum / total))
            i1 = max(i0 + 1, min(nw, round(nw * (cum + len(line)) / total)))
            t = start if k == 0 else max(sw[i0][0], t)
            d = max(0.2, (end if k == len(chunks) - 1 else sw[i1 - 1][1]) - t)
        plan.append((t, d, line, cum / total, (cum + len(line)) / total))
        cum += len(line)
        t += d
    return plan


def _norm_tok(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


def split_cues_llm(src_segs, tr_segs, wmap, max_chars: int, src_name: str, tgt_name: str):
    """The 'brain': a language model cuts every translated sentence into film-style subtitle cues (complete meaningful
    clauses) and cuts the source sentence at the same places, so the timing comes from the real word times.
    Returns a list of cues (start, end, text) per sentence, or None for a sentence it could not split safely."""
    hard = 2 * max_chars
    system = (
        f"You are a professional film subtitler. For each item you get a sentence in {src_name} (\"s\") and its translation "
        f"into {tgt_name} (\"t\"). Cut BOTH into the same number of subtitle cues, the way cinema subtitles are cut: every cue is "
        "a complete, meaningful clause or phrase that can be read on its own; never end a cue on a preposition, conjunction, "
        "article or the first half of a name, and never separate a verb from its object or a noun from its adjective. "
        f"A cue may hold at most {hard} characters of the translation (hard limit), and ideally {max_chars} or fewer when the "
        "meaning stays whole. Prefer FEWER, more meaningful cues over many tiny ones; a short sentence stays as ONE cue. "
        "Do not change, add or drop any word: concatenating the cues must give exactly the original \"s\" and exactly the "
        "original \"t\", in order. "
        'Return JSON {"r": [[{"s": "...", "t": "..."}, ...], ...]} with one list of cues per item, same order.')
    out = [None] * len(src_segs)
    todo = [i for i in range(len(src_segs)) if src_segs[i][2].strip() and tr_segs[i][2].strip()]
    for k in range(0, len(todo), 8):
        ids = todo[k:k + 8]
        try:
            res = _llm_json(system, {"items": [{"s": src_segs[i][2], "t": tr_segs[i][2]} for i in ids]}, 90).get("r")
        except Exception as e:
            log.warning("semantic cue split failed: %s", str(e)[:150])
            continue
        if not isinstance(res, list) or len(res) != len(ids):
            continue
        for i, pieces in zip(ids, res):
            try:
                a, b, _ = src_segs[i]
                sw = src_segs[i][2].split()
                tw = tr_segs[i][2].split()
                ps = [(str(x["s"]).split(), str(x["t"]).split()) for x in pieces]
                if [_norm_tok(w) for p in ps for w in p[0]] != [_norm_tok(w) for w in sw]:
                    continue
                if [w for p in ps for w in p[1]] != tw or any(not p[0] or not p[1] for p in ps):
                    continue
                if any(len(" ".join(p[1])) > hard + 4 for p in ps):
                    continue
                wl = (wmap or {}).get((a, b)) or []
                if len(wl) != len(sw):
                    wl = []
                cues, n0 = [], 0
                total = len(sw)
                for ps_s, ps_t in ps:
                    n1 = n0 + len(ps_s)
                    if wl:
                        t0, t1 = wl[n0][0], wl[n1 - 1][1]
                    else:
                        t0, t1 = a + (b - a) * n0 / total, a + (b - a) * n1 / total
                    line = " ".join(ps_t)
                    cues.append((t0, max(t1, t0 + 0.3), wrap2(line, max_chars) if len(line) > max_chars else line))
                    n0 = n1
                out[i] = cues
            except Exception:
                continue
    return out


def make_cues(segments, max_chars: int, origs=None, wmap=None):
    """Timed cues, one line each when the timing allows it, otherwise two tidy balanced lines.
    origs (optional): original-language text per segment, added as small marked lines (\\x01) under the translation."""
    cues = []
    for idx, (start, end, text) in enumerate(segments):
        words = text.split()
        if not words:
            continue
        ow = origs[idx].split() if origs and origs[idx] else []
        sw = (wmap or {}).get((start, end)) or []
        plan = _plan_cues(words, start, end, 2 * max_chars, sw)                # movie style: a cue = one phrase, max two lines
        for t, d, line, p0, p1 in plan:
            txt = wrap2(line, max_chars)
            if ow:
                oline = " ".join(ow[round(len(ow) * p0):round(len(ow) * p1)])
                if oline:
                    txt += "\n" + "\n".join("\x01" + x for x in wrap2(oline, int(max_chars * 1.4)).split("\n"))
            cues.append((t, t + d, txt))
    return cues


def dedupe_repeats(text: str) -> str:
    """'هاملن هاملن' -> 'هاملن': a word of 4+ letters repeated right after itself is almost always a recognition glitch."""
    out = []
    for w in text.split():
        if out and len(w) >= 4 and w == out[-1]:
            continue
        out.append(w)
    return " ".join(out)


def fix_overlaps(cues, gap=0.08, min_show=0.35):
    """Cues must never be on screen together: each one ends before the next starts, and is held long
    enough to read (with a small hold after the voice, because word timings are only approximate)."""
    cues = sorted(([a, b, t] for a, b, t in cues), key=lambda c: c[0])
    for i, c in enumerate(cues):
        a, b, t = c
        want = max(b + 0.35, a + min(6.0, max(1.2, 0.058 * len(t.replace("\x01", "")))))   # hold: reading speed ~17 chars/s, at least 1.2 s
        if i + 1 < len(cues):
            limit = cues[i + 1][0] - gap
            if limit - a >= 0.2:                            # a short cue is better than a late next one
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





UPLOAD_MB = 300 if LOCAL else 44   # cloud Bot API: 50MB limit; own server: much larger, but keep files sensible


def video_kbps(duration: float) -> int:
    """Video bitrate that keeps the burned file under UPLOAD_MB for this duration."""
    if duration <= 0:
        return 5000
    return max(250, min(4000 if LOCAL else 5000, int(UPLOAD_MB * 8 * 1024 / duration - 64)))


def target_res(duration: float) -> int:
    """Highest picture size (short side) that still gets enough bitrate to look good inside the upload limit."""
    kbps = video_kbps(duration)
    res = 1080 if kbps >= 2500 else 720 if kbps >= 1200 else 480 if kbps >= 550 else 360
    return min(MAX_RES, res)


def out_size(w: int, h: int, duration: float = 0.0):
    """Size of the burned video: short side from target_res (never upscaled beyond the source)."""
    short = min(w, h)
    res = min(target_res(duration), short) if short else target_res(duration)
    if short and short != res:
        f = res / short
        return int(w * f) // 2 * 2, int(h * f) // 2 * 2
    return w // 2 * 2, h // 2 * 2


def chars_per_line(w: int, h: int, job: dict, duration: float = 0.0) -> int:
    """How many characters of the chosen font fit on one subtitle line, so lines never overflow the frame."""
    ow, oh = out_size(w, h, duration)
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
    w, h = out_size(w, h, duration)
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
        margin_v = 10 if portrait else 8
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
                rtl = (job.get("bi_lang") if small else job["tgt"]) in RTL_LANGS
                if rtl:   # keeps ". ، ! ( )" at the correct (left) end of right-to-left lines
                    ln += "\u200f"
                if small:  # second-language line: smaller and softer, in a font that has its letters
                    ar2 = job.get("bi_lang") in ARABIC_SCRIPT
                    k = 1.15 if (ar2 and job["tgt"] not in ARABIC_SCRIPT) else 0.9 if (not ar2 and job["tgt"] in ARABIC_SCRIPT) else 1.0
                    ln = ("{\\fn%s\\fs%d\\1c&H00D9D9D9&}" % ("Noto Naskh Arabic" if ar2 else "Noto Sans",
                                                                  max(5, round(font_size * 0.7 * k)))) + ln
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
    # keep the file under Telegram's bot upload limit: cap the video bitrate by what fits in ~44MB for this duration
    rate = []
    if duration > 0:
        kbps = video_kbps(duration)
        rate = ["-maxrate", f"{kbps}k", "-bufsize", f"{kbps * 2}k"]
    await run_ffmpeg(
        [
            "ffmpeg", "-y", "-v", "verbose", "-threads", BURN_THREADS, "-filter_threads", BURN_THREADS, "-i", str(video), "-vf", vf,
            "-c:v", "libx264", "-preset", BURN_PRESET, "-crf", "29", *rate, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "64k", "-threads", BURN_THREADS, "-movflags", "+faststart", str(out),
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
        "max_filesize": 600 * 1024 * 1024,
    }
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        opts["cookiefile"] = COOKIES_FILE
    def _is_botcheck(err) -> bool:
        m = str(err).lower()
        return "confirm you" in m and "bot" in m

    try:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            if not _is_botcheck(e):
                raise
            # YouTube blocks datacenter IPs: try other player clients once before giving up
            log.info("youtube bot-check, retrying with other clients")
            opts2 = dict(opts, extractor_args={"youtube": {"player_client": ["tv", "android_vr", "web_safari"]}})
            with yt_dlp.YoutubeDL(opts2) as ydl:
                ydl.download([url])
    except Exception as e:
        log.warning("yt-dlp failed for %s: %s", url, e)
        if _is_botcheck(e):
            raise UserError("yt_block")
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
        if LOCAL and tg_file.file_path and os.path.isabs(tg_file.file_path):
            try:
                os.remove(tg_file.file_path)   # the own server keeps its copy otherwise
            except OSError:
                pass
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
/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAQDAwQDAwQEBAQFBQQFBwsHBwYGBw4KCggLEA4RERAOEA8SFBoWEhMYEw8QFh8XGBsb
HR0dERYgIh8cIhocHRz/2wBDAQUFBQcGBw0HBw0cEhASHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwc
HBwcHBwcHBz/wAARCAEYARgDASIAAhEBAxEB/8QAHQAAAQQDAQEAAAAAAAAAAAAAAAQGBwgBAwUCCf/EAFsQAAECBAIGBQUJCgsG
BAcAAAECAwAEBREGIQcSMUFRYRMicYGRCBQyQqEVI1JicrHB0dIkM0NFU3OCkpSyFzVjZHSToqSzwuEWJoOEo/AnNkTiJTRGVFVW
df/EABwBAAEFAQEBAAAAAAAAAAAAAAUAAwQGBwECCP/EAEIRAAEDAwAFCgUBBQgBBQAAAAEAAgMEBREGEiExQSJRYXGBkaGxwdET
FDJC8OEHFSNSchYkM2KCorLxwiVDU9Ly/9oADAMBAAIRAxEAPwCn8EEEe15RBBBCSRBBBCSRBBBCSRBBCum0yerM61JU6Ufm5t02
QywgqUe4bueyOgEnAXHOawFzjgBJIEgqUlABK1ZBIFyewb4n/Bfkzzs4G5nFM8JJo5+ZShC3jyUv0U92tE84Y0e4Zwcge49Il2Hg
LGYUOkeV2uKufC0EoLXNJtfyR4qk3XT23URLIP4rujd3+wKqLhzQvjbEwQ5L0VyVll5h+fPQJtxsrrHuESnRPJVVZK63iMJO9qQY
v/bX9mLGjaSdvGPaTBOO1QM+raqHW/tAutQSISIx0DJ7znyCi2l+TtgKnhJfp81PrG1U3NKIP6KNUQ7pDRrg2mW81wvSGyN5lUrP
iq8Oe8J56oydLlzMT00xKsD8I+4ED27YkCniZ9LQq5LeLlVOw+Z7ieGT5LMvSadKpsxT5NocG2EJ+YQrSy1a3RN2+QIj+oaZsKSC
ilqafnVj/wC1ZJT+sqwjjOafKbf3mizqxxW8hPzXj0WjmTrLRdJhrCN3bs81Kj1Jp80LPyEm6DucYQr5xHDn9GuDapfzvC9HcJ3i
VSk+KbGGQ1p7klH3yhzSR8V9B+gR1pTTZh5+wel6hLk7y0lY9hhsxg7wnhbrxT7Whw6j7FIqr5OWj+paxapszT1H1pOaWAP0Vawi
P655JgspdDxKb7mqgx/nR9mJwp+kLDNTIDVYYQo+q/do/wBoAQ5WXUPthxpxDjZ9ZCgoeIiO+jidvapMOkl8oDgyu6nDP/JUXxJo
MxzhlK3HqK5OSyMy/T1dOm3EpHWHeIjpaS2tTawUrSbKSoWKTzG6PpjrcNvGGtirR3hjGzZTW6PLTLtrCYCdR5PY4myvG8QZLaN8
Z71ard+0h4IbXRZHO32PuF89oIsZjPyV5yUDk1hOo+eNjPzKdIQ72JcHVV3hPbEBVejVCgT7shVJJ+TnG/SZfQUKHPPaOYygdLA+
I8oLRLZeqK5t1qWQE824js3pBBBBDSKIjNoxBCSRBBBCSRBBBCSRBBBCSReCCCEuoggghJIggghLiIwTaFtLpc5Wp5iQp8s5Mzj6
tVtloXUo/wDe/YIs/o00GyGFuhqddS1P1oWUhu2sxKnkPXV8Y5DcN8TKShlqnYZu4lA73pBSWeLWnOXHc0bz7DpUZaOtA9WxWlqo
VlTlKpCrKSCn7ofHxUn0R8ZXcDFnMMYOomDpHzSi09qVbI66x1nHTxWs5q+blHTCrZkwzsb6WaBgRC2Zh0zlVtdMjLqBWPlnYgdu
fKLNDRwUbdbjzlY5cLzddI5/gsBIO5jd3bz9Z8E+grVvyzjiyeN6LUq0aNTZsVCoIGs8mUHSNy6eLjnop4WuSTuiqmItIeMdK1WY
o8s4plmdcDTVOkyUIVf4atqgBmScgAcos5o8wLI6PsPNU2U1VzC7Lmpm1i+5bM8kjYBuHMmFHOZnYYNg4rzcrBHZ6cOrX5lfuYOH
S4+g3ninlaE0/UJWlyjs5OzDcvKsjWW66qyUjthHXcQyGGqU/Uqk8GpZobs1LVuSkb1HhFW8cY9qOOJ/pJhRZkG1EsSaFdVscT8J
XPwtD7jhRbFo/NdH630xjefQdPkpFxZp4eeW5K4aa6JoZeezCLrVzQg5DtVc8hEUT1SnKtMqmZ+bemphWZceWVH27O6OaiN6Iayt
Qo7VS0DdWnZjp4ntShEb240IhQ3HlSHpQiN6I0IEKG4SjPSlBtC+QqM3TnA5JTT8s5xZWU/NCBAjcmwtfZHCoUjWuGHDKkGjaV69
IlKJvoqg0NvSjUX+sn6QYkyg6RKLWdRtbpkplWXRzNgCeSth9kV+aTvtCgG0eS0FV6sstLPta3VPR7blae4tfjsjgYqwbQ8bU/zG
uU9mcZF9RShZxo8ULGaT2RDWH8ZVfD6kpYf6aVG2WeJUju3p7ol3DeNafiEJbB82nT+AcPpfJPrfPyhp0eRg7VWZ7fV254micdm4
jYR7KsGkryc6zhRL1Rw+p2r0hF1KbCfulhPNI9MDinPiIhKPpne0QrpY0BU3GgfqtDDNOxAbqWLarE2fjgeio/CHeDtgVUUH3Rdy
vmjun2SKe6dj/wD7D1Hbzqm14IXVii1DD9SmKbVJR2UnpdWq4y6LFPPmDuIyMIYFkEbCtTY9r2hzDkFEEEEcXUQQQQkkXggghLuE
QQQQksoggghLiI6eHcPVHFNWYpdLly9NvbtiUJ3qUdyRxjxRaLPYiqktTKcwX5yYVqoQMgOJJ3ADMndFv9HuAZDANHEtL6r0+8Aq
amyLF1XAcEjcO85wTt1ufWPydjRvPoFWtJNIo7PDhvKlduHqejzXrR1o4pmAKfqsasxVHk2mZ1SbKX8VPwUct+0w93HWmGXHnnEN
MtpK1uLUEpSBtJJ2COZU6vJ0OnzFQqEwiXk5dOs46s5AfSTsAGZirWkrSzP46eVJy3SSlBQrqS97KetsU5bbyTsHM5xaJpYqKMMa
OoLKbdaa7SOqdNK44zynHyHtuCeGkvTy9MrdpeE3VNMZpcqVrLX+aB9EfGOZ3W2xBSnFLWpbilLcWSpSlG5UTvJO0xk5wuw/RH8S
1+m0eXuHZ59LIUPVBPWV3C57oASzSTvy5a5b7bR2inLYW6oAyTxOOJKsX5OWB0SNLdxXONjzqdCmZMKHoMg2Usc1EW7BzicJh1qX
aceecS2y2krWtRsEpAuSeVoS06Tl6XIy0jKoDcrLNpZaQNyUiw9giLdPGLzT6VL4elnLP1AdLMkHMMg5J/SUPBPODscYhjDVi8rp
tIrsT/Me5o9h4qKtIuO38cVtTiFKRSZUlMoycst7hHwlewWENVBjSM43JENHpWv09LFSwtghGGtGAlCIUIjQ3ChEcXHLemFCI0IG
YhQiOKM9b0bIUNiNKN0Km0a2QzJ2AQlEkOFvZTrEAAknIAC94nHR/o5bpjSalV2EuTrife5dwXDKSPWG9RHh2wm0caO/ckNVerN/
d5GswwofeB8I/H+bt2SDVazLUSQdnJsr6NvYlCSpSzuSBxMdws/vd7dM/wCUoz0Ejj0BQ5j/AAUMNPidkre5j69UIJzZWc9Xmnhw
3wz0Zx28S1mp4rqBmplh5DSLhlgIVqtJ8Myd5jmIkpgH/wCXe/q1fVHko5RmRlO1tQ4Fyy2nZC1s6tiDYjYRujUiVfy94e/qz9UK
ESz/AORd/UP1RxJ72nin3hjSE9KlErVlKdl9iZnatHyvhDnt7Yk9h1uYaQ62tK21jWStJuCOIMV+ak3lW95d/UP1Q7sL12doKw0p
l5yRUes1qnqningeW+Gy3O5Vq5W6N2ZIdh5udd7SXoupGkmldBNpEvU2EnzWfQm62j8FXwkE7U94sYpDizClUwXW5ikVeXLM2zmC
M0OoOxaDvSePcbGPogxNNzbCHmVazaxcG1vZDO0l6N6bpJoSpKbszPM3VJzgTdTC/pQd6e/aBA+qpBKNZv1InoppbJapBS1RJhPe
3pHRzjtHTQWCOpiLD1RwrWZukVWXLE9Kq1Vp2hQ3KSd6SMwY5cBCCDgrcY5GyND2HIO0FEEEEcXpEEEEJdyiCCCEuIj002t91tpp
CluuKCUoSLlRJsABxJjwTE56B8Bhxf8AtXUGrpQSiQQobTsU73Zgc7nhEqjpH1Uoib/0ENu1zjtlK6pk4bhzngPzgpF0U6OmsDUn
pplCVVycSDMODPok7Q0k8BvO88gIfs7Oy1Nk35ybeQzKsILjjqzZKEjaTHlBvYCK2aY9JBxLOqolMevR5Rfvi0nKZdG/mlO7ic+E
XeV0VvgDWDqHOsboqKq0iuBfKd+1x5hzDyAXC0m6SJrHtT1GtdmiSyj5vLnIrP5RfxjuG4d8MUZQQRVpJHSOL3HaVslJSQ0cLYIG
4aPzvXoGJm8nChJn8XT9UcTdFNlSEHg44dUf2QrxiF7xZzyZ5QNYWrM5brTM8G78kNj6VmJNCzXmaEC0vqDT2iUt3uw3vO3wypr5
bIp/jivqxLi6q1DWJaW8W2eTaOqn2C/fFqcVz5peGKzOpNlS8o6tJ+Nqm3tIimbaSlIB2gQaqOAVL0CpBmapI27GjzPolKY3JhOg
woREdaE5KG4UIhO3nClEIqO9KEQpQITohS3HFFet6ByiV9DtKpE3OvzUy6lyqSxuxLLGSU/lB8I3y5bd8RSgQvkp2Yp8yzNSrq2Z
hlWshxBsUmOjYUHudM+qp3Qsdqk8VbMm8YuRvtDNwLjlnFkr0L2qzVWk3caGxwfDTy4jd2Q4q1THqpT3GJeefkZn0m32VEFKuY3j
iIcO7KyOakfBP8Cfkn829S6aVq+EfGPRWT6x8YgCp1rFlFnnJKdqs+282cx0pIUNygd4PGE6cX18/jid/rTDRKPN0Zmc0ObI0g9a
sLrH4R8YzrE+sfGIARiivLz9150/8UxvOK6zLtOurq05ZtClklzgCY8lef7NzA41x4qewo/CPjBrnifGKHjS7jpRB/2sq1jn9+/0
iy+Aa3WZ7BlDmp+ozL82/Lhxxxxd1KuSRfutEeJ4lOApl40RqLXE2WWRpycbM9ali94BDUZn5wkAzDh747ElNPIPvyypKvhbRDpY
VV5ICzeUxdM+ihnSNRBMSaEIxFIoJlXDl0ydpZUeB3HceRMUkfYdlnnGHm1tPNqKFoWLKQoGxBG4gx9KdaKy+Uro0CCca0tmwUQi
ptoG85Je+ZKu48YFV9LkfFbvG9aPoHpKYpBbKk8l30HmPN1Hh09arXBBBAdbAiCMwQkliCCME2hLqcWCMKPYyxHKUtvWSyo9JMOj
8G0n0j27hzIi4UpLMyEqzKyzSWpdhAbbbTsSkCwERloSwumh4Z9030WnarZzMZpZHoDvzV3iJDrFXlqFSZypzitWWlGy4vieCRzJ
sB2xe7LRClpviv3u2nq4LH9LLi+413y0W1rDgdJ4n0Ud6acfKoNMFDkHdWpVBBLq0nNlg5HsKswOV+UVxvlC2u1maxDWZyqTqrzE
04VkbkjckcgLAdkIAYCVlUamUv4cOpX6yWllspGxD6jtcec/puC9CMx5gvERGEHfFrfJ3sjR0CNqp98n+yIqkYtZ5Phto6R/Tpj/
ACwRtn+P2FUzTsZtf+oeqdulFwp0eYgIOZlwnxWkRVC1yYtLpUWf4Pq9be0n/ETFQ6jPFSlMtmyRkojeeEEK6VsW0oboJEXUbwP5
vQJU9VWWSUoBcUOGzxjSMQOpOTCLc1GOQbQQCdWSk7DhaCKeMDaE5JXELKiA82pv4wOsIcEu6h9AW2tK0HekxHgEK5SdekXNdlwp
O8bj2iHYq1w2P2qPNQtcMs2FSKgbIUIENymYll5khuYsw7suT1Vd+7vhxoN7cIIska8ZaUEnifGcPGEoRChAvaNCBClAj2oD0up8
09ITLUzLOqafaUFIcQbFJifcFY1YxRLdE7qtVNpN3GhkHB8NPLiN3ZFfWxCppxbSgttakLHrJJBF8jnHQ7CAXa1xV7MO2OG4/nBP
7SRiiTrU21JSbTbolFHWm95O9KT8Hid52Q06SwxMVGWZmVKSw6sIUpBsRfIHxtHNWHG2HCy2HHEpJQi9tY2yF90RRUscVmeKm0u+
ZN7ChjJQ7VbfmhiWYM2uRC12Vz4fl6c4DeJ37eP5sVs2MB01IsXZq/yk/VDQ0tUin4WwLVJxt5/p3kiWZClDNazbhuTrHuhxaMMb
NY0wpLTKnAalKgMTiN4cAyX2KGfbcbohfygMbIrtdYoMo4FSdJJ6Yg5KmCLK/VHV7SqPU0jRFrDigVmoq+a7immccRnLuoe+zvUP
MMuTLzUuykqeeWltCRvUo2HtMX2o+HZSk0yRkQFKEowhnNWXVSB9EVV0EYSViLHctOut60jRx524SMivY2n9bP8ARi3utuhiijIa
XHipf7QbhrzxUkZ+kZPWd3gPFZQ020OogDmIFG4g1k62prDXtrat87Xte3C8a1qETMLOsE710JOY10dGo9ZOzmI9zknL1GTmJObZ
S9KzLamnWljJaFCxB7o5SHC2sLTtSbx2m1pcQlafRULiPDmJt2Y3BzdioLpKwO/o+xdO0dzWXKg9LKPK/CsK9E9ozSeYMNGLk+UV
ggYnwWqrSzWtUqHd8aozWwfvie6wUPknjFNorNXB8GQtG7gvobRW8/ve3tmceW3ku6xx7RtRBBBEVWTCI7eD8PqxTiWn0sX6N5y7
qh6rac1nwFu8RwzE2aA6JqoqlbcTmoiUZJ4ZKWf3RE+20vzVSyI7uPUEKvVd8lRSTDfjA6zsHupvbShptKG0hDaAEpSNiQMgPCIV
074sJ8zw4w5kLTU1Y/1aT7VeETK8+3LsOvvK1WWkFa1HckC5PgIp3X6w9iGuT9VevrzbqnAD6qdiR3JAEXK91HwYRE3e7yCz3RG3
/MVZqH7meZ3epSLbBHkHdE+YD8n6TxThGnVmpVWdlJieBdQyy2ggN3sknWzuQL94irxQvmOGBX253WltkQlqnYBONxO3sUC60egb
xZpvyXqGpQBxFVAN56FrKK7V2jv4frNQpU0m0xJPrYXzKTa/eLHvj3LTSQ7XhRrZf6G6Ocykfkt2nYR5rn7BFqfJ/VbR2j+mv/Om
KrHIRaXQCq2jxv8Apr/zpiXats/YUH04H/pn+oeqcOll3otHOIV70sAj9dMU21iSeMXB0vm+jTEX5lP+IiKfDKO3o/xWjoTGgDcU
Mh/z+gXoZx3cP4MxBirpfcWjzk+lrJxbLd0pPAqNhfltjgiLzaLJWXp2jvDTUs2ltC5Jt5QA9JaxrKUeZJiDR03zDiCdgRXSi/Ps
tM2WNms5xwM7hxVKKlTJ2jTjslUJR+UnGjZbL6Chae4wjveL6YqwZQscyQla5IofCQQ0+nqvM/IWMx2ZjlFZNIegWuYT6WdpGvWK
Qm6ippH3QyPjoHpDmnwEOVFBJFtbtCg2LTWjuOIZ/wCHJzHceo+h8VEpN46tKr83TCEpPSMD8Gs7Ow7o5KcxHuIbHuactKuMkbZB
quGQpQpFalKqAGl6r1s2l5K7uPdHbSIhdKyghSSQoG4INiIc9HxvMSxS1PpU+1s6Qemnt4/PBGKtB2SIBV2pwy6Hb0KSUQoQNkIK
dOy1RYD0s8l1s70nZ2jdHRbicCCMhV2UFpLTsKUNbYjLSNhwyE37qy6PuWZV76B+Dc49ivnvEooTlfcM78IZOLcb09mWmKdLIbnn
XUltZObSO/1j2eMM1GrqYcpFqfM2pBhGefqTFwviyr4QnXpukTXQOvtKZXcayVJOw2O8HMHcY46UOzD6UIDj0w8sAJHWUtSj7SSf
bGCQkchxhXR6tOUOqStTkHehnZVYcac1QrVPYcjAsHcCVczEGl0sbRrkb+fG7JVx9FOCxgTCbMk8lJqUyennFjP3wjJF+CRl23O+
O3jHEzeD8Nz9aclHpsSqbhpoZkk2GsfVTfadwiIMLeUlJONIZxJTXGHQLGZkRroPMtk3HcTEgSulfBVYaKEYhp4S4CktzJLZIO0E
LAgyySMs1WOwsQuFpubK01FdA54LsnG0EcwIzjZsHMqvu6T8SqxgcUieIqV9XVt710X5LV+By79ucWn0fY/p+kGk+dyhDU21YTUo
pV1Mq+lJ3K+mK46VMFUKizYqeGqvTpmmzK+tJszSFrllH4IBuUH+zs2Whn4YxLUcI1diq0t8tTLWRBzS4nehQ3pPD6YgsmdC8h+0
LQbjY6O+0LJaVuo9o5OzGMfaR+Y3jZvvWuwhbTZnWC2idnWEMHA+kOnY/pPnUp7zNtWEzKKVdTKvpSdx+mHVLPFmZbXuvY9kFRh7
ctWQ1VHLTvdDO3Dm8E4HEIebW24gLaWClaFbFJIsQe0RQPSHhJeCMYVaikHoZd3Wl1H1mVdZB/VNu0GL/bIrl5VGGguWomJGkdZC
jIzCgNxupsnvCx3iBNyg14tcbwrT+z+5mluPyzjyZRjtG0eo7VWaCCCK8tzygkAEnYItVgKk+4WEaTJkWd6EOu/LX1j89u6KzYfp
/utXabIWuJmYbbPYVC/svFuNYeqLDcOUW/RWny6SY8NnqfRUXTSc6sVOOknyHqmbparppOCptlCrPVBaZVPHVOa/7II74rSYlLTn
VS/WqbTUq6kqwXlD4yzYexPtiLAbwzepviVTm8G7ER0ZpPgUDXcX7fbwXTw3QnsTYhplHYv0k6+lokeqknrK7kgnui+smhmUlWZV
hIRLsIS22keqlIsB4ARWrycsMiYqlTxC8i6JNHmrBP5Reaj3JFv0osW5Mtyza3XVBDTaStajsSkC5PhEu10+rEZDx8lQtO601Na2
lZtEY/3H9MLp60Vf8pLDnufiaSrrSLM1Rro3SB+GbAHtRq/qmJk0a4+ax9QX6glKW3WZp1lTY3Jvds96CnvBhNpgw2cVYDqTDSNe
clB55LgbSpFyQO1OsIeqYxUU5czrCFWGWWzXhrJ9m3Vd1H9cFU2Ji0ugNVtHqB/PX/8ALFWNovxi0OgVVtH6P6Y/86YG2cf3jsK0
PTYZtv8AqHqu/pdVfRtiEfyKf8REVC2ExbjS0r/w4xB+ZT/iIipBjt7GJm9XqUzoIMUMn9Z8gsE2EXk0duf7hYY//nMfuCKMmLs6
PndXAuGhwp7P7gjlmbmR3Uon7Q261JD/AFHyT3S5G9tZve+ccxt2++FKHLQccxY++NMjHmhfDmOQ7NNoFLrKs/O5ZA1XD/KI2K7R
Y8zFX8b6NcQYBmLVSU1pNSrNzrF1MOfpeqeSrGLvIcjMwyxOyzktMstvy7ydVxp1IUhY4EHIwMqKBkm0bCrTZNMa214jkPxI+Y7x
1H0OxfO8nOC8WZx95N0pOh2fwg6iTmD1jTn1HoV/IWc0dhuOYiutXodSw/UHJCqyT8nON+k08nVNuI3EcxlASankhOHha/aL9RXZ
mtTP28WnY4dnqNi0yM9M0+YD8q8tpwb0nbyI3iHknSTNJlUp8xYMyNrhUdU89X/WGNsjGtHGTPj2NKIT0cM5BkbnC61UxRVasCia
m1dCfwTfUR4Db3wlptMnKzONydPlXZqadyQ0ynWUfqHPZCVtTaXW1Ot9I2lQKkaxTrjeLjZ2xZ/RdjPBM3KNU2hSDNEn1Czkq8bu
PK4h05udhseUSKdgnfh7sIVeK99qpviU8Bd1bh0nj4doSLRroTlqGW6niNDM5UdrcoQFssc1blq9g57YMYeT9Taopybw7MJpsyrM
yrt1MKPxSM0e0dkS5cp27oC/B35KIM1MLJzpJcjUmqEpyeH245sbvXpVNsQ6O8T4aWoT9GmehT+HYT0rR/STf22hrZA2UQCNxyi+
iHM7gkHlCSoStMcaW9PScittCSpbkwyghIG0kkbIhvtYzlrlaKXT6TAbPBk9Bx4EHzVGBq3sCm/KAmJF0m4+p2Ipo06g06SlKQwu
5falkNuTSh61wLhHAb9p3CGxhPCNRxnV26fTkZnrOvKHUYRvUo/MNpOUDHR8vUYcq+w1hNN8zUt+GMZIJ3Dp9l1dFUhiGcxfKrw4
50Mwz1n31glpDROYcG8HZq7Sdmy4uGpWW7uhtYSwpTsF0dum05HVHWdeV6b696lfQNgGUdsrg7SUphZgnasf0ku7brVa8bcNbsB4
npPoOCdUu6XZdpe9SRDR0r0D/afR5iCnpRrPiXL7I/lG+un90jvhx0tzWkW+RI9sLUhJ9MAoOSgd43x5liDgWHiqpTzupKlk7N7H
A9xyvm+DcAjYRcQR18VUk0DE9ZpRFvMpt1kD4oUbey0EUtwLSQV9QRSCVjZG7iM9672iWU86xvJLIuJZtx/sITYe1UWOQq+UQToS
Y1q1VJj8nLJQP0l/+2Jwbc1FBR2DMxoejMWpQ63OSfT0Wc6Uu+JXavMAPX1VadJE6ahjitO3ult7oE9iAE/ODDW1tXPhCqfmTOT8
1MqNy86tw96ifphxaNsO/wC0mMqbLOI1pVlXnL/DURnbvOqO+Kq8OqKg6u9x8yrsHsoqPWfuY3yCs/o0ogwvgqlU9SQmZUjp5j84
vrEdwsO6ONpuxIaHgSaYaXqzVUV5oixz1Tm4f1Rb9KHkl+2Zit2nPE3u1ixNOaXeXpTfRWBy6VVlL8OqO6LPX4paXVb1BZVYqR1z
uwmlGcEvPfkeOFu8n3ExpGLnqS4q0vVmtVIOwPIupPiNYeEWjDuy4BHA74odITr9Ln5WellFMzKupebI+Ek3HzRdikVlmtUqSqMu
QWJtpLyOQUL27jl3REs0muwxHgp+nNu1KllY0bHjB6x+nkqk6RsODC2MqpTkJIluk6aX/NL6yfC5HdE56CTq4BR/TH/8scTygsO+
c0+nV9pF1yqvNXyPgKN0E9irj9KOxoOVbATf9Mf/AMsco6f4Ne5nDBwpt0rvn9H45T9WQD1jPnv7V29LCr6OcQfmU/4iYqZeLW6V
lf8Ah3Xx/Ip/xExVDO8Rb8MTt6vUoloQMUT/AOr0CCcjFzsAuf7k4cH8wZ/dEUxzi4eA3LYLw8OEiz+6I9WEZlf1JnTxutSxf1Hy
T0bd5wqbd5xyW3YVId5xYnxrJ5Il1kORvS5HLbe2QqQ5EZ7FDfGuglcc3EeF6LjCnmRrcg1Nsj0FKFnGjxQsZpPZClDkbkr5xHfG
CMELxG+SB4kicWuG4jYVVfSD5PNZoBdnsOqcq9MF1FkJ+6Wh8kZLHNOfKIYUkoJSoEKSbEEWIPAx9FUOWtnDEx7ofw7pBC5h9oyN
XIyn5ZIC1H46dix2584EVFu+6LuWjWP9oD48Q3MZH8439o49Y7iqSjOPYOrYjIjOHzjzRNiLR8pTs9LiZpl7IqEsCpo/K3oPJXcT
DDJvApzXMOHDBWpUtXBWRCaneHNPEKUMIabazQC1KVXXqlOTYXWr39sclH0uxXiIsRQq5JYjpjFRpz4flHh1VDIg7wRuI3iKT2iR
9D+NF4XxE3JPukUupLDTqSckOHJDg78jyPKCdDXOa4MkOR5Ko6SaMQTwuqaRuq8bSBuPPs5/NWlCrRxMXYfZxdh+cpD8w9LomALO
tHNKgbi49ZN9oO2OkpyxIjz0l4sLow4Fp3FZdC98L2ysOHNOR1qpiNGOIV4sOHPNdWZT11TBB6ENX++629PLbfLbFnsHYUp+CqMi
nyCSSes8+odd9fwlfQNgEdYL5QdJESnoWQOLhtKOXjSGqukbYn8lo3gcTz+wW8rjwVxqLkeSuJmFX9VOajH7h/TVHR1t0cykdWnt
fGJV7YXa0RXDaUKl+sqm3lBU0U/SjVHEpsmdaZmhzKkBJ9qDBDn8qKTDeJ6DOAff5FbZPEocP0LgimVzNSoeOlfRejE5ntFO8/yg
d2z0XG0JNgN1t3iplHsUYlGfe6Gnzbl/QZcV4JMRloWIFPrHHp2/3TEi1jOjVIDb5s7+4Y0OyDFsYRzHzKqd6brXJ+eceQVVWldR
N+EWD0FUMSVEnay4kdLPr6JonaG0HPxVf9WK+ycq7OPS8qwnWffUltA4qJAHzxbuhyTVFpUlTmfvUo0loc7DM95ue+AWj9N8WYyn
c3zKL6WVWpTCnbvedvUP1wuhW6s3Q6PPVJ4+9yjKnSOJAyHebDvinUxNOzk0/MzCyp99anHFHepRufaYtvXaNKYmpblOnlPCVcUl
Sgy5qFWqbgX4X+aGcrQrhQnJNQ/av9IJ3a31FU5ojxgdPFAtHrlR21jzNnWdzDgFXcZxYnQJXzO4em6O6u7tOc12wfyS7n2Kv4iM
t6FcKA5pqH7V/pDgwzo+omEagqephnEvqbLSg4/rJUk22i3IREoLXVU0we7GOO1Sb5e6C40joG51t42cR29ic+IqW1iGg1GlPW1J
tlTYJ9VW1Ku5QBhiaF0OyuC1S76Sh5memELSdygQCPEQ/lPa2+EMjItSCpssjVTMzCphQ+MoC/iRfvg26mzO2UcAQqjDOW0j6U7i
QR1jYfzoXC0qLvo9r35pP+ImKr8YtFpQVfR/XfzSf8RMVdMVnSAYqG9XqVftDRijf/V6BYMW5wM5bB1AH8yZ/dEVGOyLXYJcthCg
j+ZNfuiHdHBmV/V6rxpkzWp4+s+SeSHYUId5xykOQpQ5FocxZlJCuoh3nCpt2OShzZClt3nEdzFBkiXXQ7eN6XI5SHecKUOxGdGo
j4l00uRuQ5aOchyN6HLwy5iiuYliyl9tbTiUraWClSFi6VDgQdoisOnzRZSMKy0tiKiMiUl5h/oJiUSfe0qUCUrQPVBsQU7NlrRY
Sv4mpWFqeqoVeealJUZBSzms8EpGajyEVW0vaXlaRFS8hIy65aiyjhdR0v3x9drBagMkgAmyeefITcXRBha76uCumg9JcfnmzU+R
F9x+0jm6T5KMAAYwslIJSbKGYI3GMA2jayw5OOoYZSVPPKDaEjaVE2A8TAIbTsW2EgDJVxKJPqqFGp02v035ZpxXaUAmF+vHMp8q
KbT5STBuJZlDN/kpA+iFWvF8Y0hoysEma0yOLd2ThKdeDX5wm14Okj1qpr4aUlceC4Re0aC5CqmM+dzzKNqQdZXYI4RjavLhqgkp
4SqOglmW96UgHtjdeNZVeMBcQ8IEdpyq+eVKzdnCz+8LmG/Yg/RBCjyoiDSMM/0p/wDcTBFOuoxVO7PJb3oQ4myw5/zf8imFoZds
3WW/jNL9ihEoTaemkppr4bK0+KSIh3Q/M9HV6kxf75LpV+qr/wB0TAHMxw3xf9HD8S2sHWPEoXfY9Wuc7qPgFC2iChe6OJPPnE3Z
preuPzisk+HWPdE+JXawhm6PqH/s7RXWnE6sxMzDjq+NtYhA/VAPfCrHVbNCwtPzSFaswtPQs/LXkD3C57oct1OKCi1pBjeSol0c
64VuqzoaPzrQdKuFWXFtrqh1kKKTZhZFwbbQI9jSvhI5+6h/Z3PsxWcCwtHoHKK6dIqk/aPH3R3+ylGd7nd49lZf+FnCI/Giv2dz
7MYOl7Ce6pqH/LufVFaSYxaOf2gqf5R4+68/2Rov5nd49lbyiV+RxDIJnqc+H5VSigL1Sk3BzBBzEdPXiDtB9cDM1UKK4rqvp85Z
BPrJyUO8WP6MTQXOcWe31PzcDZePHrVOuduFFUuhG7h1Jt6Tl3wFXPzSf8RMVhvmYsvpLXfAlb/NJ/fTFZ73isaRjFQ3q9Srjok3
FK/+r0CDnFpcGOWwnQx/M2v3RFWos3g5z/dWif0Rr90Q9owMzSdXqu6Us1oI+v0Ttbcz2wpQ5zjlIchU25bfFuexZ9LCuohyFKHI
5aHIUIdiO5iHSRLqIdhQh3KOWhznDTxfpSouDkrZW555UwMpRhQuk/HVsT7TyiJO9kLdeQ4Caht81VIIoGlzjzKRlTKGGluuuJba
bGspa1AJSOJJyAiH8beUJI0vpJPDLaKhNjIzjgPQIPxRtWfAdsQtjHSLXcaOFM9MdFIg3RJMXS0nhfeo8z7IacVeruxfyYNg5+Kv
1o0FhiIluB1j/KN3bz+XWutXcRVTE8+qeq069OTSstd05JHBI2JHIRy7R5BtHQpVJnq5OtyVOlXZqac9FppNz2ngOZygPynu5yVf
QIqePAw1o7AEhMS7oawM8++MV1BookZUkSSVj7+9s1x8VOZvvNuEOnBGgSXlFtTuKHETLwsUyDKvek/LV63YLDmYkWrzra3US0uE
olpcaiUoACcssgNw2CD1ttLy8STDGOHuqJfNKY5w6joDnP1O4Y5hz55+5auk55QdJzhIHYz0kWbVVL+ElfSRjXhN0kY6SFqrnw0p
6SHPh2W6KWVMrHWeyT8kfWYblLk1VKaS2LhtPWWRuH+sPcWQkJSLJAsAN0MS7NiGV8gaPhjfxWitVZih0afqcyoJYk2FvL52GQ7z
Yd8KZZxbkrLrdFnFtoUoDcSkE+2IQ044t85m6RguTXd2emGXJvVOxBWAhB7T1uwDjE5LICilOwZCIDJA+RzRw815qrcaajhmeMOk
yR/SMAd+3wVf/Kgf+58Ls7yuYc9iB9MEcXymp0O4ioUoDfoJNbhHDXct/kgioXU5q3/nBbVodEY7NADxye9xUc6NpzzTF0mCerMI
Wz4puPaBE6g37IrVSpw06qSU4D94eQ53A5+y8WRCwsApN0nMHiIueh82tTSRcxz3j9FH0ih/jNk5x5JQlVtkRHpervnFQk6ShXUl
U9M4B8NWwdyf3olN19Eqy4+8rVZaSVrPBIFz7IrZVqi5V6pNz7vpzLhctwB2DuFhErSWq+FTiEb3eQUexUuvOZSPp8ykl4IxeMxR
Vbkb4zHmPUJdXSoNYcoNZkai3e8s6FkD1k7FDvBMWlZmUPtIdbVrNuJCkqG8EXB8IqOdkWE0YVB2fwbJdKbql1LlweKUnL2G3dFn
0aqD8R8B3Hb3Kq6TUgexk43jYl+khd8DVofySf30xW7jFi9Iqv8Acisj+TT++mK5nOGNJhipb/T6lP6MNxTOH+b0CDFlMILtheij
+aNfuxWoxY7CarYZo/8ARG/3Yf0UGZ5Or1XvSJuYmdfonMhcKm3I5qF3jLs61JtLefdQ0y2LqWtQSlI5kxdHtAGSqTJDnYF2UOWh
HWsT0zDcp5zUptDCD6Cdq3DwSnaYi7E2mNplK5agNh1zYZt1PUHyUnb2nLkYiSoVCaqs25Nzkw7MTK/SccVc/wCg5CKtcL7FFllP
yjz8P1Rah0XkqCH1PJbzcT7KQMX6Y6tWteVpOvTZE5FaT7+4Oah6I5DxiOdcm9zcnMk741wXioT1MtQ7Xldkq70lDT0bPhwNAHn1
lbNsY1SSkJBKlGwAFyTwh2YO0eVrGK0rlmfN5C9lTj4Ib/RG1R7PERYbB2jag4PSh1hnzqpAZzkwAVg/EGxA7M+cS6O1zVPKxhvO
fRCLtpJSW7LM67+Yep4efQogwXoRqtd6ObrSl0uQOYaI+6HB8k+h2nPlFgsNYapOE5ISlJk25ds+moZrcPFSjmqFutbOOHU8RoaC
mZNQU5sLo2J7OJi0Ultig+gZPOs1uF1r7y/VecN5huHXz9q6NbrIl0mXZV7+oWUR6g+uGyHLQh6YkkkkqOZJO2PaXIKNjDQvcNG2
FuqEtDkZ14R68Z6SFqr2Y0rLnOPcu07NvIZaTrLUchw5xpkpZ+feDTCNZW8nYkcSYe1MprNNZsjrOq9Nwjby5CG3uDetQquobA3A
+pK6bJt06WDSM1HNa/hGEOL8VSeDsPzVWnTrBsarTV7F5w+igdu/gATG6o1SVpEi/PTz6GJSXQVuOrOSQP8AvZvip+kXSFM4+q/T
WWzTJa6ZSXJ9Eb1q+MrfwFhAWvqhA3/Mdy5o9Y5LvU68n+G08o8/QOk+AW7ASpvGGlilTs8svTD04Zx9W7qArsOAGqAOVot6HOMV
x8nWj+cV6q1Zabok5cMIPx3Dn/ZSfGLDrdS2CtZs2kXUeAG2GbXHiEvdxKmabSiS4Np4xsjaBjr2+WFU3TrUxUdJNSSk3TJttSo7
UoufaowQx69VVVyu1OpLJJnJlx7uUokey0EUypk+LM5/OSthtlN8rRxQfytA8FzzsPCJ8wXU/dTDVPeKruIR0Lnyk5fNY98QHEja
K6t0b07TFnJwB9scxkoeFj3Qf0Vq/gV3wzueMdu8e3aot5g+LT6w3t2p7Y5S89hGrIl1EOdFrG20oBBUPAGK+iLMqSl1CkLTrIWC
lQO8HaIrtXKWuiVeckF394cISeKdqT4EQc0qpyHMnG7d6qBY5AA6LtSGC8YEZioo8sxmMCC8cXEHOJ20TK1cIIA3TLvziIJiYtEE
+F0aflNbrsTHSAclJH0pMH9HHAVoB4goRfGF9KeghOfSGSvBVYAGYaB8Fpiu94svWpT3VpE9I75llbYPMjL22is+otClIcSUuIJS
pJ2gjaIkaVRlszH8CMdx/VR9HXARPZxBz+dyNpifsCViTqOHJBpp9HnEqylp1oqAUkgWvbgdt4gMC0GYsYEWq5ut8heG5BGMIpXU
Qq2BpOMKdcSaQ6dh/WaZUJ2eH4JpXVSfjK3dguYiLEOKaniZ4LnnyW0m6GG+q2jsHHmbmOTfKPNrR7uN5qK04ccN5h6868Udtgpd
oGXc5/NiwDaNgVHlptb7qGmkKcdWdVKEAlSjwAG2JTwhodmpxTc1X1qlWNolGz76r5R2J7Bc9kQqWkmqnasTc+Scra+noma87seZ
6gmDRcPVLEc2JSmSjkw9tOrklA4qUckjtibMH6GKdSiiarikVCcGYYA94Qezas9uXKJBpVPkqNJok6fKtS0snYhtNrniTtJ5mF2v
Fso7HFDypeU7wWfXXSapqsxwchvie3h2d63NhLaUoQkJSkWAAsAOAG6NE7WJamp99Xdzc2nNR+rvji1Cdqrt25OSebR+UUBrHsF8
o4fuTUlkqVKulRzJNiT7YNhg4oHBb2P5czwOjO1dCoV+YqN0k9Ex+TSdvad8c9K49po9RH/pHfZG5NFqKshKLHaQPphwYG5FG/Ai
GqwgDrC1Bce0uQsZw7UFkawbbHxl3+aOpLYWbBBmJhS/itiw8THkuaFGlq6dm92epcJKitQSkEqOQAFyY7tPw6++QuaPQt/B9c/V
HclJOWkhZhlKD8IZk98Kwu8MukPBCai5OdsiGOle5VlmTaDTDYQgbhtPad8a6tW5Cg052oVKaRLSjXpLXvO4AbSTwEczEtYXQaBU
ao3LKmlSbKnehSrV1rc/b3RVHE+MavjCe86qkxrBF+iYRk2yOCR9JzMB6+ubTDGMuKlWLR6S7PMj3YYDtPE9A913tJWkmcx3OBlo
LlqMwq7MtfNZ+GvirgNg9sMO9o9XjqYboL2J6/T6Sze826EKUPVRtUruSCYqz3vnkydpK1mCGnt9PqRjVYwf9lWW0I0Y0bAcq64n
VfqS1Tar7dU5I/sgHvjraU657g4CrU0lWq8615s1x13Or8xJ7ocUu03KsNMMpCGWkhCEjckCwHgIgvyiMShx2k4faX97BnHwDvN0
oB7tY94ix1jhR0RxvAx2lZJbInXm9te4bHO1j1DbjyCgcAJFhsGUEerQRQVuaxC2jVNdHqspPIuSwsFQ+EnYod4vCKCPccjo3h7N
42ry5oc0tO4qyDDyHmkOtKCm3EhSVDeDmDEdaVaGXG5esNJzbsw/bh6qvG47xCzRtXfO6cqmOq9/lM27n0myfoOXeIes1Ks1GUfl
JhOuw+goWOR+mNb/AId4twLfuHc4fr4KnhrqKp28PJVtjIhfW6S9QqpMSD+a2lWCty0nYodojngxm8jHRuLHDBCtLXB4DhuK9RmP
N4zHldwsmHPo/rwoeIWumVqys2OgcJ2JueqruPzmGvBq3yh6nndTytlZvBym5ohLGY3birQg3yiMNImBnOldrdNbK0OdaZZSLlJ3
rA3g7/GOho8xm3UGGqVPuhM82NVlxZ+/JGwX+EPb2xIJXq7I0OSOnvFLv394Kp7TNbp8/wDRCq8SLbYxeJ3rGj+iVp1TymFS0wvM
uSx1LniU7D4Rw/4GJQq1hWJnU4dCm/jeKjNo1WsdhgDhz5x5o7He6YjL8g9SiQG3ZDzwro5quJ9R8p8zp5/9Q8nNQ+InartyHOJI
omjig0dxLpYXOPpzC5o6wB4hIy+eHq25awvlBCh0YIOvVHsHqfZDa7SA41aUdp9B7rl4bwZScKo+4WdaZIsuZd6ziu/cOQhxpXzh
r4jxtScLtkTTpemyLplWiCs9u5I7Yiap6WMRTcypyVeakmfVZbbSuw5qULk+EFJ7hR28CJvDgOCAxWmtuLjK7jxd+forDJcI3xvS
5FZhpRxWPxoP6hv6o9p0qYsH40H9Q39mIZ0hpj9rvD3Th0Tqz9ze8+ys0Fx6CorMNK2Lf/yg/Z2/sx7GlfFw/Gg/Z2/sx5/f9N/K
fD3TR0QrD9ze8+ys0lce0risg0tYuH4zR3yzf1Rk6W8XKFvdRKeaZdsH5o8m/U/MfD3TZ0OrT9ze8+ys/rQa/OKxUzS7imnzwmH5
8zzRyXLzCRqEctUApPMe2JlwjpLo2LdRltwylRO2UfIBJ+IrYr5+USKW6U9QdUHB5ihlx0brKJuuRrN5xtx18U+kr5xsCoTAkdse
wqCOFXyxbnmW5yXelns2X0KbWD8FQsfYYphVJFdKqM3Iu5OSrq2VdqSR9EXKK8orRplpXubjZ6ZSmzVRaTMD5Xoq9qQe+AN9hzE2
QcDjvVy0KqPh1MlOdzhkdY/Q+CYF7RPPk/4Z1G53Eb6M13lZW43fhFDvsnuMQIo5Hbblti5uHJeTkqFTGKcLSKJdvoeaSkEE8ze5
5mBtmpxLMXn7Ub0zrnU9GIWf+4cE9A39/lldeammpSXemH1hthlCnHFn1UgXJ8IpdievvYoxDUau9cGbdKkJPqIGSE9yQInfTpjA
U2hN0GXXabqXWesc0sA7P0lC3YDFcjEfSKr15BTt3N39f6JvQa1/BgdWvG1+wdQ9z5Igggitq+oggghLqXUeqPUWpS86xmtpWady
0nanvETzIzzM/KMzcurWYeSFpPL690V4MPXR/if3OmfcyaXaVmFXaUTk2s7uw/P2xbNFrsKWb5aU8h/gf1Qm6UnxmfEbvHknPpCw
2a5TxOSyLz8mkkAbXG9pT2jaO/jENpNxFkwcxxERVpAwh7nPLqsk39xPKu8hI+9LO/5J9hg5pJaif75CP6h6+6h2yq1f4L+z2TDj
N4xBFMRtehGRHkRmOJL0FFJuCQRmCIfNA0nz0glDFTbVOsJyDoNnQOd8ld9jzhiRkAxIpa2elfrwuwmZqaOdurIMqfaZjrD9QSki
pNsrPqTALZHjl7Y7Ir1KCdb3UktXj5wj64rX6OUYuL3sPCD7NKZwMPYCe0IQ+wxOPJcQrAVLH+Hqakk1Ft9Y9SWBcJ8MvbEf4g0q
1CoJUxSmzIsHIukgukctye655wwc1CMap4RCq7/V1A1WnVHR7p6ns1NCdYjWPT7L2palqUpa1KWo3KlG5J4kxgx5G2DbAUHnRUbF
6yggCTwgzEJJZEZvHm5jIhLi9Rgm0YN4MzCSRtj2nqkEZEG4PCMBJGZEZJtHQub1JmEdMVUooblasFVKRFgFKPv7Y5KPpdh8Ymuh
4qpWJZXzilzaHkj00bHG+Sk7R80VH1o3SU/NUybbm5KYdl5lv0XGlaqh/pygvR3iWDDZOU3xVZuejFNV5kh5D+jceseyuR0l4i/T
jRhN4ekqmlN3JF/UUfiOC37wT4xysHaaG19HK4jbCFbBOsp6p+WgbO1PhEmV2VlsUYVn5aXeafYnpdQZdbUFJKrXSQR8YCD75oa+
nc2I5JG7iqdDSVNmro5J24AO/gRuO3qVS72iyWi/FsqNG4mZ54Ibouuy8o7QhPWR3kKAHMRWhSjchQIUMiOBhWzU5tinTNPbfWmT
mXEOutDYtSL6pPZc+zhFPpa80cheBnYR7eK0S8Whl0gbC44wQc9HHwSvE9fmcUV6dq01k5MK6qL5NoGSUDsH0xyYIIDve57i9xyS
jUUbIoxGwYA2BEEEEeE4iCCCEuIjyRHqAwklLGBcViqsJp82v7uZT1VK/DJH+Yb+O3jD0Wlt5pbLqEuNOJKVIULhQO0GK7MPuSzz
bzK1NutqCkrSbFJG8RMOFMXN4gl+ieKW6g0PfEDILHwk/SN0aRo3fW1LBR1J5Y3E8Rzdfmq/cKExn4se7yUf4zwi5h2YL8uFOUx1
XUWcy0T6qvoO/thqRYt9lqbYcYfbS6y4kpWhYuFDgYiHF2CXqEtc3Jhb1NJud6meSuI5+MQ75YnQE1FOMs4jm/TyT1FXB+I5Dt80
0zlEvYUwVRHqHJvzUqmafmG0uqWtRsL52AB2DZEP3vHZpeKqzR5cS8nPKQwDcIUlKgnsuMoF2mqpqaUuqWawI2bjjvUishklZqxO
wVM7WA8Nq/FDB71fXCoYEwyE/wATy/ir64iFGkXEiBlUf+ij6oFaSMTE/wAZD+pR9UHTebZ/8P8Atb7oQ62Vp3SeJTl0l4OpdHpj
NSpzQllB5LS2kqJSsEGxAOwi3hDZ0fUCUxLiVqTnlHzZDS3lISrVLmrbq377m24RyqtiGp19baqjOLfDXoJsEpTzAGV+cJZObfp8
01NSry2Jho6yHEGxSYr9TU00lYJo48M2bOfnRKKmqG0hhc/lnODzKyrGjrClgDQ5U96vrhUjRthE7aDKeK/tRBQ0pYrRkmq/9Bv7
MexpYxeNlW/6Df2YLG6287ovAKtPsV1O6f8A3OU8o0ZYOO2gSniv7UKW9F2DD/8AT0n4r+1EADS1jEfjf/oN/Zj1/C9jMbKx/d2v
sw065UPCPwCiP0dvJ3VH+53srDt6LMFHbhyT8V/ahUjRNghW3Dcke9f2oreNMeNk7K1/d2vsxvRprxwkfx1/dmvsw0bhRndH4BR3
aM3w7qn/AHu9lY8aIcC2/wDLUl4r+1FbtL2EZDBOMVyFMWrzJ9hEyhpStZTOsSCm+0jq3F87GN38N2OiCPdv+7NfZhj1OpTtZn35
+oTTs1OPq1nHnVXUo/8Ae6IVXUQSNxG3BRew2a6UVQZKyfXbjGMk7e3dhTnoH0a4dxTQ52sVqWTPuiZVLNy61kIaCUpJJAIuTrb9
wiY29DOAja+F5HxX9qKe4YxrXsGuPuUSpOynT26RACVIXbYSlQIuOO2HUjTzj5P49/urX2Y5FUQNYGubt7FAvGj16qqx81NU4Ydw
1nDHRgDCs8dDOAP/ANWkf1nPtQzNKGhfB7GCaxUaXTm6ZPU6XVMtusuK1VaouUqBJBB2cQbRCytPOkA/j7+6s/Zji4k0p4vxXT1U
6rVp16RWQpbKG0NpXbZraoFxfOxyjslRAWkBnkolDozfoahkklXyQQTynHZxGCMFNFIyEZtGAq0SFgHRpMYnLc/UNeWpANwdi5jk
ngn43hEanp5Kh4jjGStBqqqKkjMsxwB+bFH+yOvQMXVjC0x01Mm1NpJutlXWac7U/SLHnEpYr0Ny0xrzGH3BKu7fNXVEtq+Sranv
uOyIeqlKnqLOLlKhLOS8ynMoWNo3EcQeIh2qpamgcHOGOYj3UWkrqO6RljcHnad/cvFQm0z9Qm5wMpYEw6p3okG6UFRuQL7rmEu0
wGAbYDyv1iSUYY0NAaOCzBBBEde0QQQQkkQQQQl1EYjMEJJY2xsl5h2UfbfYcU282dZK0nMGPEYjrXFpBacFcIzsKl3CmMWa4lMv
MarVQSM0bEu80/VDsFlAggEEWIOYMV3SotqSpKiFJNwQbEGJFwxpBCgiUrCrK2Jmrbfl/X48Y0Sx6UNlAp644dwdwPXzHp3IFW20
t5cI2c3stmJtGqHyubooSlw5qlCbA/IO7sOXCI1fl3ZV5bL7a2nUGykLFiDzEWKQ6ChK0qCkKFwpJuCO2OTXqBT8Rs6k4z76kWQ8
jJxPYd45HKJty0bjnzLS8k83A+3kmae4Oj5Mu0eKgYnON0lJu1Cdl5RgAvPrDaAchc8YcNcwDVKRrOMJM7KDPXaT10jmnb3i8cCR
mnqfOMTTJAeYWFpJF7EHfFLlpZKaUMqWkc/V0Iw2VsjMxnKkyW0RMltPS1dzXtnqMC3dcwo/gglSP44f/qE/XCEaXQEJ1qR19+q/
YX5dWMp0wAfidR/5gfZi0g2HH/6QV7blwPklw0Myyvxy/wD1Cfrjc3oTlVH+O3/2dP1wjRpkSPxMr9pH2Y2p01IT+JV/tI+zDbv3
Fw/8lFe28faf+KXo0HSivx5Mfs6ftRvRoIk1fj2Y/Zk/ajnp04oT+I1/tI+zGxOnhKfxEs/80Psw0TZeH/koj2X/AO0/8V0hoBkl
D+P5kH+jJ+1ETYtwzNYRrsxS5laHS2ErQ6gWDiFC4Nt3ZxESanT8kfiBZ7ZofZiMsVYlmsW1p+pzaUIccASltHotoAsEjj277mBl
x+Q1B8r9Wend2qfZ23cTH548jHRv7E6dG+it7HUrMT79Q8ykGnCykpb11uLABNgSAALjOJDR5N0goZ4jmv2VH2ojnR9pSmcCSkxJ
Kkkzsi650qUdJ0am12ANjY3BsMuUPdHlIpSf/La7f0wfYjtMbeIh8X6uO9Dbq3SI1TzRn+Hwxq+u1LH/ACapYtr6HEj/AEtur0ko
nVvzsq9uyIIq1LmaHVZ2mziQmak3VMuAG41gdx4b++JyX5SieiV0WGvfbdXpJy6b87JvaIPqtRma5VZuozaukm511TrhSNqlHYB7
AIjVopSB8vv7fVTtHheQ9/70PJ4fTnP+nh1pGFRvlJOYn5huWlmXHn3DZDbadZSjyEO3DmjCr1kodnQadJnPWdT74ofFRu7TaJow
1hul4Xly1T5cJdULOPr6zjnarhyFhEuhss9Rh0g1W9O/sCm198gpgWx8p3Ru7SmXg7RC3JqbnsQBDz4spEkDdCPln1jyGXbEqJOo
AAAAMgBuEa1upSlS1qCUJF1KUbADiTEUY10spbDkjh5YW56K562SfzY3n4xy4cYsb30dohydnmfzuVQ+HW3mfn8h+d5TpxxpGlMK
tLlZfUmawodVralm/rL+hO08hFfajUJmqzrs5OvrfmXjrLcWcz9Q5QnWtbri3HFKW4slSlKNyonaSd8eYolyustc/LtjRuH5xV7t
dnht7MM2uO8/nBEZjEZgWTlFkQQQR5SRBBBCXUQQQQkkQQQQkkQQQQkkRixjMEJJduhYsqFAOo2rpZS+cu4cv0T6piTaFiun1wJS
w70cydrDuSu74XdELWjIuCCMiDcEbosFq0jqqDDDy2cx9Dw8lBqaCOfbuPOrFI6pB3xyqxhikVsKVNSiQ+fwzXUX4jb33iNqPpBq
dMCWpgidlxlZ09cDkr67w9qZjmkVQpSX/Nnj+DmOr4K2GLzTXu23Jvw5CATwd+YQWWinpzrN7wmlVdGE40pS6dNNzCNzb3UX47D7
Iak7QKpTCfPJCYaSPXKLpPeMonxCgoBQIKTsIzBjalRAyORhup0XpZeVES3xH52rsdzmZseMqt45G8F4sDOUOl1G5madKuk+sWwD
4jOOO9o6w8+SRLOtE/knlD57wGl0VqW/4bwe8KW27R/c0hQsYIlxeiyjrzRNzqP0kn6I8DRPTT+MZz9VH1REdo3XD7R3hOfvWn5z
3KJo9gxLrWiijJIK5ueX+klP+WOlLaN8Oy5BVKuvEflXlG/cLR6ZozWnfgdvsm3XinG7J7FCJsNuUdKn4aq9WI8yp0w6k+vqaqB+
kbCJ5kKHSaaQZWmyjSh6waBPic46/S6wAO6CEOivGaTuHqfZQZr6R/hM71EtG0RzbpSuqzzbCNpalxrr/WOQ9sSXQcK0XDwSqRkk
B8D7+713PE7O60LVKCEFSiEpGZJNgO+GvVtI9DpGslMwZx8fg5brC/NWwe2CjaS321uu7A6Sdv51IVJPW1x1Bk9A3fnWnuTrZ74b
mIscUnDKVImXulnLZSrJBX37k98RVXdKFZq6VMyyhT5U5arB98UOa9vhaGUolRJJuSbknaYA3DSpoyyjGek+g91Po9GiTrVRwOYe
6c2KMeVXFJLTq/N5C90yrR6p+UfWPblyhsQQRTJ6iWoeZJXZJVqggjgYI4m4CLxiMwQynkQQQQlxEEEEJdRBBBCSyiMwQQlxEYgg
hJIggghJIggghJIggghJIjFoIISSXSFZqFKN5OceZHwUq6p7jlDmktJVTZsJliXmQN4BbV7MvZBBBClulZS7IZCBzZ2dx2JmSmil
+toKcEnpNpqxaZlZplR3pssfQfZHWl8dUF+3/wAQS2eDqFJ+iCCD1LpZXk6r9U9Y9iFAltcG8ZCXt4mozg6tVkz/AMUD543ivUu1
/dOTt+fT9cEEWSK9TvbktHj7oZJRRg4yV5XiiitenVZMf8UH5oRPY+w8xe9RS4RuabUr6IIIHVukdVD9LW9x909Da4ZN5P52LlTe
lWltAiWlJt9XFQDY9tz7Ib87pVqzwKZSXlpVJ9axcV7cvZBBFbn0juE2z4mqOgY/XxRGO00rPtz1pp1KuVKsKJnp1+YHwVr6o/RG
Xsjn29kEEBZJHyO1nkk9KntY1gw0YCzBBBHhekQQQQkkQQQQkkQQQQkkQQQQkkQQQQkl/9k=
"""   # round bot logo (280px JPEG), used in the welcome picture and result cards


# ------------------------------------------------- saved user settings + glossary
PREFS_FILE = Path(os.environ.get("DATA_DIR", tempfile.gettempdir())) / "subvid_prefs.json"
PREFS: dict = {}          # str(user id) -> {"ui","tgt","mode","font_i","size_i","style_i","pos_i","bi","gl":{wrong: right}}
_prefs_lock = threading.Lock()


def _clean_db_url(url: str) -> str:
    """Fixes typical phone-typing damage: spaces/quotes, 'Postgresql://' capital letter, and an '@' inside the password."""
    url = "".join(url.split()).strip("'\"")
    m = re.match(r"(?i)^(postgres(?:ql)?)://(.*)$", url)
    if not m:
        return url
    rest = m.group(2)
    if "@" in rest:
        cred, hostpart = rest.rsplit("@", 1)
        if ":" in cred:
            user, pw = cred.split(":", 1)
            from urllib.parse import quote, unquote
            cred = user + ":" + quote(unquote(pw), safe="")
        rest = cred + "@" + hostpart
    return "postgresql://" + rest


DATABASE_URL = _clean_db_url(os.environ.get("DATABASE_URL", ""))
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "5"))        # finished jobs per person per day (the admin is exempt)
ALLOWED_ENV = {int(x) for x in re.findall(r"\d+", os.environ.get("ALLOWED_IDS", ""))}
ALLOWED_DB: set = set()
DB_OK = False
DB_STATUS = "غير مفعّلة (الحفظ في ملف مؤقت)"
DIRTY: set = set()
ACTIVE: set = set()       # users with a job running right now
MEM_JOBS: dict = {}       # (uid, day) -> finished jobs, used when there is no database
_RIYADH = timezone(timedelta(hours=3))


def _today() -> str:
    return datetime.now(_RIYADH).strftime("%Y-%m-%d")


def _db():
    import psycopg   # imported here so the bot still starts if the library or database is missing
    return psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=8)


def _url_shape(url: str) -> str:
    """Describes the connection string without revealing the password, to find typos."""
    try:
        from urllib.parse import urlsplit, unquote
        u = urlsplit(url)
        pw = unquote(u.password or "")
        odd = [c for c in pw if not c.isalnum()]
        return (f"scheme={u.scheme} user={u.username!r} host={u.hostname} port={u.port} db={u.path!r} "
                f"password_len={len(pw)} starts_with_bracket={pw.startswith('[')} non_alnum_in_password={len(odd)} "
                f"has_whitespace={any(ch.isspace() for ch in url)}")
    except Exception as e:
        return f"unparsable ({type(e).__name__}): url_len={len(url)} starts={url[:11]!r}"


def init_db():
    global DB_OK, DB_STATUS
    if not DATABASE_URL:
        return
    try:
        with _db() as c:
            c.execute("create table if not exists prefs (uid text primary key, data text not null, updated timestamptz default now())")
            c.execute("create table if not exists jobs (id bigserial primary key, ts timestamptz default now(), uid text, ok boolean, secs integer)")
            c.execute("create table if not exists allowed (uid bigint primary key)")
            for uid, data in c.execute("select uid, data from prefs").fetchall():
                PREFS[uid] = json.loads(data)
            ALLOWED_DB.update(r[0] for r in c.execute("select uid from allowed").fetchall())
        DB_OK, DB_STATUS = True, f"متصلة ✅ ({len(PREFS)} مستخدم محفوظ)"
        log.info("database connected: %d users, %d allowed", len(PREFS), len(ALLOWED_DB))
    except Exception as e:
        _m = str(e)[:120] if "://" not in str(e) else "(hidden)"
        DB_STATUS = f"فشل الاتصال ❌ ({type(e).__name__}: {_m}) - الحفظ في ملف مؤقت"
        log.warning("database unavailable, using the local file: %s", str(e)[:200])
        log.warning("DATABASE_URL shape (password hidden): %s", _url_shape(DATABASE_URL))


def load_prefs():
    try:
        PREFS.update(json.loads(PREFS_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    init_db()
    log.info("prefs loaded for %d users", len(PREFS))


def _flush_prefs():
    global DB_OK
    with _prefs_lock:
        uids = list(DIRTY)
        DIRTY.clear()
        try:
            if DB_OK:
                with _db() as c:
                    for u in uids:
                        c.execute("insert into prefs (uid, data) values (%s, %s) on conflict (uid) do update "
                                  "set data = excluded.data, updated = now()", (u, json.dumps(PREFS.get(u, {}), ensure_ascii=False)))
            else:
                tmp = PREFS_FILE.with_suffix(".tmp")
                tmp.write_text(json.dumps(PREFS, ensure_ascii=False), encoding="utf-8")
                tmp.replace(PREFS_FILE)
        except Exception as e:
            DIRTY.update(uids)
            log.warning("could not save prefs: %s", str(e)[:200])


def save_prefs():
    threading.Thread(target=_flush_prefs, daemon=True).start()


def uprefs(uid) -> dict:
    DIRTY.add(str(uid))
    return PREFS.setdefault(str(uid), {})


def record_job(uid, ok: bool, secs: int):
    """Counts a job for the daily limit and the /stats numbers."""
    if not uid:
        return
    if ok:
        MEM_JOBS[(str(uid), _today())] = MEM_JOBS.get((str(uid), _today()), 0) + 1

    def run():
        try:
            if DB_OK:
                with _db() as c:
                    c.execute("insert into jobs (uid, ok, secs) values (%s, %s, %s)", (str(uid), ok, secs))
        except Exception as e:
            log.warning("could not record job: %s", str(e)[:150])
    threading.Thread(target=run, daemon=True).start()


def jobs_today(uid) -> int:
    if DB_OK:
        try:
            with _db() as c:
                return c.execute("select count(*) from jobs where uid = %s and ok and ts >= "
                                 "(date_trunc('day', now() at time zone 'Asia/Riyadh') at time zone 'Asia/Riyadh')",
                                 (str(uid),)).fetchone()[0]
        except Exception as e:
            log.warning("could not count jobs: %s", str(e)[:150])
    return MEM_JOBS.get((str(uid), _today()), 0)


def is_allowed(uid) -> bool:
    if uid == ADMIN_ID:
        return True
    pool = ALLOWED_ENV | ALLOWED_DB
    return not pool or uid in pool


def stats_text() -> str:
    lines = [f"📊 الإحصائيات\nقاعدة البيانات: {DB_STATUS}", f"المستخدمون المحفوظون: {len(PREFS)}",
             f"الحد اليومي: {DAILY_LIMIT} · المسموح لهم: {len(ALLOWED_ENV | ALLOWED_DB) or 'الجميع'}"]
    if DB_OK:
        try:
            with _db() as c:
                day = "(date_trunc('day', now() at time zone 'Asia/Riyadh') at time zone 'Asia/Riyadh')"
                t = c.execute(f"select count(*), count(*) filter (where ok), count(distinct uid) from jobs where ts >= {day}").fetchone()
                w = c.execute("select count(*), count(distinct uid) from jobs where ts >= now() - interval '7 days'").fetchone()
                al = c.execute("select count(*) from jobs").fetchone()[0]
            lines += [f"اليوم: {t[0]} طلب ({t[1]} نجح، {t[0] - t[1]} فشل) من {t[2]} شخص",
                      f"آخر ٧ أيام: {w[0]} طلب من {w[1]} شخص", f"الإجمالي: {al} طلب"]
        except Exception as e:
            lines.append(f"تعذّر قراءة السجل: {str(e)[:100]}")
    return "\n".join(lines)


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


# --------------------------------------------- export / edit / re-burn text
KEPT: dict = {}            # job id -> saved result (video + subtitles) so the user can edit the text and re-burn
KEPT_TTL = 2 * 3600
KEPT_MAX = 3


def _drop_kept(jid):
    st = KEPT.pop(jid, None)
    if st:
        shutil.rmtree(st["dir"], ignore_errors=True)


def keep_state(job, workdir, video, w, h, duration, cues, lang, chat_id) -> bool:
    now = time.time()
    for jid, st in list(KEPT.items()):
        if now - st["ts"] > KEPT_TTL or st["chat"] == chat_id:
            _drop_kept(jid)
    while len(KEPT) >= KEPT_MAX:
        _drop_kept(min(KEPT, key=lambda k: KEPT[k]["ts"]))
    KEPT[job["id"]] = {"dir": workdir, "video": video, "job": dict(job), "w": w, "h": h, "duration": duration,
                       "cues": cues, "lang": lang, "chat": chat_id, "ts": now}
    return True


def latest_kept(chat_id):
    now = time.time()
    for jid, st in list(KEPT.items()):
        if now - st["ts"] > KEPT_TTL:
            _drop_kept(jid)
    mine = [(st["ts"], jid) for jid, st in KEPT.items() if st["chat"] == chat_id]
    return max(mine)[1] if mine else None


async def offer_text_edit(bot, chat_id, ui, jid):
    try:
        await bot.send_message(chat_id, T(ui, "tx_offer"),
                               reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(T(ui, "tx_btn"), callback_data=f"tx|{jid}")]]))
    except Exception as e:
        log.warning("edit offer not sent: %s", e)


def export_text(cues) -> str:
    """Numbered entries; the bilingual original lines are shown with a leading '| ' (the translation is the main text)."""
    return "\n\n".join(f"{i}\n" + "\n".join(("| " + ln[1:]) if ln.startswith("\x01") else ln for ln in t.split("\n"))
                        for i, (_, _, t) in enumerate(cues, 1))


def _split_marks(t: str):
    main = [ln for ln in t.split("\n") if not ln.startswith("\x01")]
    small = [ln for ln in t.split("\n") if ln.startswith("\x01")]
    return main, small


_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_edit(text: str):
    """Numbered blocks (optionally with SRT times) -> [(start|None, end|None, text)]."""
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "").strip("\ufeff\n ")):
        lines = [ln.rstrip() for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        if lines[0].strip().isdigit():
            lines = lines[1:]
        start = end = None
        if lines and (m := _TIME_RE.search(lines[0])):
            g = [int(x.ljust(3, "0")[:3]) if i in (3, 7) else int(x) for i, x in enumerate(m.groups())]
            start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
            end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
            lines = lines[1:]
        lines = [("\x01" + ln.lstrip()[1:].strip()) if ln.lstrip().startswith("|") else ln for ln in lines]
        if lines:
            out.append((start, end, "\n".join(lines)))
    return out


async def send_export(bot, chat_id, ui, jid):
    st = KEPT.get(jid)
    if not st or st["chat"] != chat_id:
        await bot.send_message(chat_id, T(ui, "tx_expired"))
        return
    st["ts"] = time.time()
    body = export_text(st["cues"])
    if len(body) <= 3500:
        await bot.send_message(chat_id, body)
    path = st["dir"] / "subtitles_edit.txt"
    path.write_text(body, encoding="utf-8")
    with open(path, "rb") as f:
        await bot.send_document(chat_id, f, filename="subtitles.txt", caption=T(ui, "tx_how"))


async def apply_edit(msg, ui, text: str):
    chat_id = msg.chat_id
    jid = latest_kept(chat_id)
    if not jid:
        await msg.reply_text(T(ui, "tx_expired"))
        return
    st = KEPT[jid]
    edits = parse_edit(text)
    old = st["cues"]
    if not edits:
        await msg.reply_text(T(ui, "send_media"))
        return
    max_chars = chars_per_line(st["w"], st["h"], st["job"], st["duration"])
    if all(a is not None for a, _, _ in edits):          # full SRT with its own timing
        cues = []
        for a, b, t in edits:
            main, small = _split_marks(t)
            main = main if len(main) > 1 else [wrap2(" ".join(main), max_chars)]
            cues.append((a, b, "\n".join(main + small)))
    elif len(edits) == len(old):                          # same numbering: keep the original timing
        cues = []
        for (a, b, oldt), (_, _, t) in zip(old, edits):
            main, small = _split_marks(t)
            if not small:                                   # user removed/omitted the original lines -> keep the old ones
                small = [ln for ln in oldt.split("\n") if ln.startswith("\x01")]
            main = main if len(main) > 1 else [wrap2(" ".join(main), max_chars)]
            cues.append((a, b, "\n".join(main + small)))
    else:
        await msg.reply_text(T(ui, "tx_mismatch", n=len(edits), m=len(old)))
        return
    status = await msg.reply_text(T(ui, "queued"))
    asyncio.get_running_loop().create_task(reburn(msg.get_bot(), chat_id, jid, cues, status))


async def reburn(bot, chat_id, jid, cues, status):
    st = KEPT.get(jid)
    if not st:
        await set_status(status, T("ar", "tx_expired"))
        return
    job, ui = st["job"], st["job"]["ui"]
    prog = Progress(ui, status, "rb")
    try:
        async with JOB_SEM:
            wd = st["dir"]
            fixed = fix_overlaps(cues)
            (wd / "subs.srt").write_text(build_srt(fixed), encoding="utf-8")
            (wd / "subs_tagged.srt").write_text(build_srt(fixed, keep_marks=True), encoding="utf-8")
            await prog.step("burn")
            out = await burn_subtitles(st["video"], "subs_tagged.srt", wd, st["w"], st["h"], job, st["duration"], prog.update)
            if out.stat().st_size > TG_UPLOAD_LIMIT:
                with open(wd / "subs.srt", "rb") as f:
                    await bot.send_document(chat_id, f, filename="subtitles.srt", caption=T(ui, "too_big"))
            else:
                await prog.step("up")
                thumb = wd / "thumb.jpg"
                try:
                    await run_cmd(["ffmpeg", "-y", "-ss", str(min(1.0, (st["duration"] or 2) / 2)), "-i", str(out),
                                   "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "5", str(thumb)], timeout=60)
                except Exception:
                    thumb = None
                ow, oh = out_size(st["w"], st["h"], st["duration"])
                with open(out, "rb") as f:
                    tf = open(thumb, "rb") if thumb and thumb.exists() else None
                    try:
                        await bot.send_video(chat_id, f, supports_streaming=True, caption=T(ui, "caption_edit"),
                                             width=ow or None, height=oh or None, duration=int(st["duration"]) or None,
                                             thumbnail=tf)
                    finally:
                        if tf:
                            tf.close()
            st["cues"], st["ts"] = fixed, time.time()
        try:
            await status.delete()
        except TelegramError:
            pass
        await offer_text_edit(bot, chat_id, ui, jid)
    except UserError as e:
        await set_status(status, "⚠️ " + T(ui, e.key, **e.kw))
    except Exception as e:
        log.exception("re-burn failed")
        await set_status(status, "❌ " + T(ui, "generic"))
        await notify_admin(bot, f"❌ فشل إعادة الدمج {jid}\nالخطأ: {type(e).__name__}: {str(e)[:300]}", key="reburn")
    finally:
        gc.collect()


class Progress:
    """One status message that shows a checklist (and a bar while burning). Edits are throttled."""

    def __init__(self, ui, msg, mode):
        self.ui, self.msg = ui, msg
        self.steps = (["burn", "up"] if mode == "rb" else
                      ["dl", "audio", "tr", "trans", "burn", "up"] if mode == "sub"
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


RETRYABLE = {"busy", "dl_fail", "timeout"}
RETRY: OrderedDict = OrderedDict()


async def fail_with_retry(status, ui, text: str, job: dict):
    """Error message with a 'Try again' button that re-runs the same request."""
    RETRY[job["id"]] = job
    while len(RETRY) > 100:
        RETRY.popitem(last=False)
    try:
        await status.edit_text(text, reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton(T(ui, "retry"), callback_data=f"rt|{job['id']}")]]))
    except TelegramError:
        pass


async def process(bot, chat_id: int, job: dict, status):
    ui, mode, tgt = job["ui"], job["mode"], job["tgt"]
    workdir = Path(tempfile.mkdtemp(prefix="subvid_"))
    st = Stages()
    prog = Progress(ui, status, mode)
    kept = False
    done_ok = False
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
            audio = workdir / "audio.ogg"
            await run_cmd(
                ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                 "-af", "highpass=f=80,lowpass=f=7600,afftdn=nf=-28,dynaudnorm=f=150:g=15",
                 "-c:a", "libopus", "-b:a", "48k", "-application", "audio", str(audio)]
            )
            st.mark("audio")
            if audio.stat().st_size > 24 * 1024 * 1024:
                raise UserError("too_long", m=max(1, int(limit * 24 * 1024 * 1024 / audio.stat().st_size)))

            await prog.step("tr")
            words_out: dict = {}
            segments, lang, alt = await asyncio.to_thread(transcribe, str(audio), job["src"], "", words_out)
            dbg = bool(ADMIN_ID and job.get("uid") == ADMIN_ID)   # /debug report only for the owner's own videos
            raw_segments = list(segments)
            vad_regions = await asyncio.to_thread(speech_regions, str(audio)) if dbg else []
            st.mark("transcribe")
            if DEBUG_TEXT:   # transcripts are private: only logged when DEBUG_TEXT=1 is set on Render
                log.info("raw transcript (%s): %s", lang, " | ".join(t for _, _, t in segments)[:600])
            if not segments:
                raise UserError("no_speech")

            names = {c: n for n, c in LANG_NAME_TO_CODE.items()}
            await prog.step("trans")
            gl = await asyncio.to_thread(auto_fixes, segments, alt, names.get(lang, lang))   # automatic proofreading
            segments = apply_glossary(segments, gl)
            segments, words_out = merge_sentences(segments, words_out)   # translate and show whole sentences
            src_segments = segments
            job["lang"] = lang
            orig_texts = [" ".join(t.split()) for _, _, t in segments]
            if lang != tgt or GROQ_API_KEY:
                segments = await asyncio.to_thread(translate_segments, segments, lang, tgt, alt, gl)
                if lang != tgt and len(segments) == len(orig_texts):   # reviewer pass
                    reviewed = await asyncio.to_thread(review_translation, orig_texts, [t for _, _, t in segments], alt,
                                                       names.get(lang, lang), names.get(tgt, tgt))
                    segments = [(a, b, r) for (a, b, _), r in zip(segments, reviewed)]
                st.mark("translate")
            if time.time() - LLM_LIMITED_AT < 120:
                await notify_admin(bot, "⚠️ حصة Groq للترجمة المجانية وصلت حدّها (429). البوت يستخدم نموذجًا احتياطيًا.", key="groq-llm-429")

            keep_joiners = tgt in ("fa", "ur")
            segments = [(s, e, dedupe_repeats(clean_text(t, keep_joiners))) for s, e, t in segments]
            if tgt in ARABIC_SCRIPT:
                raw_odd = odd_chars(" ".join(t for _, _, t in segments))
                segments = [(s, e, normalize_ar(t)) for s, e, t in segments]
                log.info("subtitle text: odd chars before=[%s] after=[%s]", raw_odd,
                         odd_chars(" ".join(t for _, _, t in segments)))
            origs = None
            bi = job.get("bi")
            if bi and mode == "sub" and len(orig_texts) == len(segments):
                other = lang if bi in (True, "src") else bi       # second language: the original, or any language chosen
                job["bi_lang"] = other
                try:
                    if other == tgt:
                        origs = None                                # same as the main line: nothing to add
                    elif other == lang:
                        origs = list(orig_texts)
                    else:
                        sec = await asyncio.to_thread(translate_segments, src_segments, lang, other, alt, gl)
                        sec_t = [t for _, _, t in sec]
                        if len(sec_t) == len(orig_texts):
                            names2 = {c: n for n, c in LANG_NAME_TO_CODE.items()}
                            sec_t = await asyncio.to_thread(review_translation, orig_texts, sec_t, alt,
                                                            names2.get(lang, lang), names2.get(other, other))
                            origs = [normalize_ar(clean_text(x)) if other in ARABIC_SCRIPT else clean_text(x) for x in sec_t]
                except Exception as e:
                    log.warning("bilingual second line skipped: %s", str(e)[:150])
            if origs:
                keep = [i for i, x in enumerate(segments) if x[2]]
                origs = [origs[i] for i in keep]
            segments = [x for x in segments if x[2]]
            if not segments:
                raise UserError("no_speech")

            portrait = bool(w and h and h > w)
            mult = SIZES[job["size_i"]][1]
            max_chars = chars_per_line(w, h, job, duration) if mode == "sub" else 40
            planned = []
            sem = None
            if mode == "sub" and not origs and lang != tgt and GROQ_API_KEY and len(segments) == len(src_segments):
                sem = await asyncio.to_thread(split_cues_llm, src_segments, segments, words_out, max_chars,
                                              names.get(lang, lang), names.get(tgt, tgt))
            for k, seg in enumerate(segments):
                if sem and sem[k]:
                    planned += sem[k]
                else:
                    planned += make_cues([seg], max_chars, [origs[k]] if origs else None, words_out)
            cues = fix_overlaps(planned)
            if dbg:
                LAST_DEBUG.clear()
                LAST_DEBUG.update(dict(lang=lang, tgt=tgt, duration=duration, size=(w, h), delay=SUB_DELAY, max_chars=max_chars,
                                       alt=alt, raw=raw_segments, words=dict(words_out), src=list(src_segments),
                                       final=list(segments), planned=planned, cues=cues, vad=vad_regions))
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
                    ow, oh = out_size(w, h, duration)
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
                if out.stat().st_size <= TG_UPLOAD_LIMIT:
                    kept = keep_state(job, workdir, video, w, h, duration, cues, lang, chat_id)
                fonts_used = fonts_for(tgt)
                await send_result_card(bot, chat_id, ui, [
                    (T(ui, "card_from"), LANGS.get(lang, lang)), (T(ui, "card_to"), LANGS.get(tgt, tgt)),
                    (T(ui, "card_len"), _fmt_time(duration)), (T(ui, "card_time"), _fmt_time(time.monotonic() - st.t0)),
                    (T(ui, "card_count"), len(cues)), (T(ui, "card_font"), fonts_used[job["font_i"] % len(fonts_used)])])
                if kept:
                    await offer_text_edit(bot, chat_id, ui, job["id"])
        done_ok = True
        try:
            await status.delete()
        except TelegramError:
            pass
    except UserError as e:
        if e.key in RETRYABLE:
            await fail_with_retry(status, ui, "⚠️ " + T(ui, e.key, **e.kw), job)
        else:
            await set_status(status, "⚠️ " + T(ui, e.key, **e.kw))
        if e.key == "busy":
            await notify_admin(bot, "⚠️ حصة Groq للتفريغ الصوتي المجانية وصلت حدّها (429). المستخدمون يرون «الخدمة مشغولة».", key="groq-asr-429")
    except Exception as e:
        log.exception("job failed")
        await fail_with_retry(status, ui, "❌ " + T(ui, "generic"), job)
        await notify_admin(bot, f"❌ فشل طلب {job['id']}\nالمرحلة: {prog.cur}\n{mode}: {job.get('src') or 'auto'} ← {tgt}\n"
                                f"الخطأ: {type(e).__name__}: {str(e)[:400]}", key=f"job-{type(e).__name__}")
    finally:
        log.info("job %s (%s -> %s, %s) timings: %s", job["id"], job.get("src") or "auto",
                 tgt, mode, st.summary())
        if job["id"] in PENDING:
            PENDING.remove(job["id"])
        ACTIVE.discard(job.get("uid"))
        record_job(job.get("uid"), done_ok, int(time.monotonic() - st.t0))
        if not kept:
            shutil.rmtree(workdir, ignore_errors=True)
        gc.collect()


# ---------------------------------------------------------------- screens
def btn(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text, callback_data=data)


def screen_ui_picker():
    rows = [[btn("العربية", "ui|ar"), btn("English", "ui|en")]]
    intro = ("🎬 SubVid Ai\n\n"
             "أرسل أي فيديو، وسيستخرج البوت كلامه ويترجمه ويكتب الترجمة عليه كما في الأفلام.\n\n"
             "Send any video; the bot will transcribe it, translate it, and add the subtitles on it, just like in films.\n\n")
    return intro + T("ar", "pick_ui"), InlineKeyboardMarkup(rows)


def nav_row(job, back):
    """Back (to a named screen) next to Cancel, so a mis-tap never throws the upload away."""
    ui, key = job["ui"], job["id"]
    row = []
    if back:
        row.append(btn(T(ui, "back"), f"bk|{key}|{back}"))
    row.append(btn(T(ui, "cancel"), f"x|{key}"))
    return row


def screen_src(job):
    ui, key = job["ui"], job["id"]
    text = f"{T(ui, 'src_q')}\n\n{T(ui, 'src_note')}"
    rows = [
        [btn(T(ui, "fast"), f"fs|{key}")],
        [btn(T(ui, "src_auto"), f"src|{key}|auto")],
        [btn(T(ui, "src_manual"), f"src|{key}|pick")],
        *([[btn(T(ui, "quick", tgt=LANGS.get(job["tgt"], job["tgt"])), f"qk|{key}")]] if job.get("saved") else []),
        [btn(T(ui, "cancel"), f"x|{key}")],
    ]
    return text, InlineKeyboardMarkup(rows)


def screen_langs(job, codes, page, title_key, list_prefix, pick_prefix, back=None):
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
    rows.append(nav_row(job, back))
    return T(ui, title_key), InlineKeyboardMarkup(rows)


def screen_src_list(job, page=0):
    return screen_langs(job, ordered(SOURCES, "en"), page, "pick_src", "sl", "ss", back="src")


def screen_tgt(job, page=0):
    return screen_langs(job, ordered(TARGETS, job["ui"]), page, "tgt_q", "tgl", "tg", back="src")


def screen_mode(job):
    ui, key = job["ui"], job["id"]
    rows = [
        [btn(T(ui, "mode_burn"), f"md|{key}|sub")],
        [btn(T(ui, "mode_txt"), f"md|{key}|txt")],
        nav_row(job, "tgt"),
    ]
    return T(ui, "mode_q"), InlineKeyboardMarkup(rows)


def bi_label(job) -> str:
    v, ui = job.get("bi"), job["ui"]
    if not v:
        return T(ui, "bi_off")
    return T(ui, "bi_src") if v in (True, "src") else LANGS.get(v, str(v))


def screen_bi(job, page=0):
    ui, key = job["ui"], job["id"]
    codes = ordered(TARGETS, ui)
    rows = [[btn(("✅ " if not job.get("bi") else "") + T(ui, "bi_off"), f"bp|{key}|off"),
             btn(("✅ " if job.get("bi") in (True, "src") else "") + T(ui, "bi_src"), f"bp|{key}|src")]]
    chunk = codes[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    row = []
    for c in chunk:
        row.append(btn(("✅ " if job.get("bi") == c else "") + LANGS[c], f"bp|{key}|{c}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    nav = []
    if page > 0:
        nav.append(btn("◀", f"bil|{key}|{page - 1}"))
    if (page + 1) * PAGE_SIZE < len(codes):
        nav.append(btn("▶", f"bil|{key}|{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([btn(T(ui, "bi_back"), f"bk|{key}")])
    return T(ui, "bi_pick"), InlineKeyboardMarkup(rows)


def screen_settings(job):
    ui, key = job["ui"], job["id"]
    fonts = fonts_for(job["tgt"])
    font = fonts[job["font_i"] % len(fonts)]
    rows = [
        [btn(T(ui, "preview_all"), f"fp|{key}")],
        [btn(f"{T(ui, 'font')}: {font}", f"cy|{key}|font")],
        [btn(f"{T(ui, 'size')}: {T(ui, SIZES[job['size_i']][0])}", f"cy|{key}|size")],
        [btn(f"{T(ui, 'style')}: {T(ui, 'style_' + STYLES[job['style_i']])}", f"cy|{key}|style")],
        [btn(f"{T(ui, 'pos')}: {T(ui, POSITIONS[job['pos_i']][0])}", f"cy|{key}|pos")],
        [btn(f"🌐 {T(ui, 'bi')}: {bi_label(job)}", f"bi|{key}")],
        [btn(T(ui, "preview"), f"sp|{key}")],
        [btn(T(ui, "start"), f"go|{key}")],
        nav_row(job, "mode"),
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
        "bi": ("src" if pr.get("bi") is True else (pr.get("bi") or False)), "saved": bool(pr.get("tgt")),
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


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    ui = get_ui(context, update.effective_user)
    await send_banner(msg)
    if not ui:
        text, markup = screen_ui_picker()
        await msg.reply_text(text, reply_markup=markup)
    else:
        await msg.reply_text(T(ui, "welcome").split("\n\n", 1)[1])   # same text without the "language set" line


async def cmd_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text, markup = screen_ui_picker()
    await update.effective_message.reply_text(text, reply_markup=markup)


async def on_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if time.time() - START_TIME < 150:
        try:
            await msg.reply_text(T(get_ui(context, update.effective_user) or "ar", "waking"))
        except TelegramError:
            pass
    if not is_allowed(update.effective_user.id):
        await msg.reply_text(T(get_ui(context, update.effective_user) or "ar", "not_allowed", id=update.effective_user.id))
        return
    kind = data = None
    if msg.video:
        kind, data = "file", msg.video.file_id
    elif msg.document and (msg.document.mime_type or "").startswith("video/"):
        kind, data = "file", msg.document.file_id
    elif msg.document and (msg.document.file_name or "").lower().endswith((".txt", ".srt")):
        ui = get_ui(context, update.effective_user) or "ar"
        if msg.document.file_size and msg.document.file_size > 2_000_000:
            return
        raw = await (await msg.get_bot().get_file(msg.document.file_id)).download_as_bytearray()
        await apply_edit(msg, ui, bytes(raw).decode("utf-8-sig", errors="replace"))
        return
    elif msg.text:
        m = URL_RE.search(msg.text)
        if m:
            kind, data = "url", m.group(0)
        elif latest_kept(msg.chat_id) and msg.text.lstrip()[:1].isdigit():
            await apply_edit(msg, get_ui(context, update.effective_user) or "ar", msg.text)
            return

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
    uid = job.get("uid")
    if uid and uid != ADMIN_ID:
        if uid in ACTIVE:
            await safe_edit(q, T(job["ui"], "busy_user"))
            return
        if await asyncio.to_thread(jobs_today, uid) >= DAILY_LIMIT:
            await safe_edit(q, T(job["ui"], "limit_day", n=DAILY_LIMIT))
            return
    if uid:
        ACTIVE.add(uid)
    JOBS.pop(job["id"], None)
    if job.get("uid"):
        # quick start must not overwrite the person's saved custom settings: only remember the language
        keys = ("tgt",) if job.get("fast") else ("tgt", "mode", "font_i", "size_i", "style_i", "pos_i", "bi")
        uprefs(job["uid"]).update({k: job[k] for k in keys})
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

    if act == "rt":
        job = RETRY.pop(parts[1] if len(parts) > 1 else "", None)
        if not job or job.get("uid") not in (None, q.from_user.id):
            await safe_edit(q, T(context.user_data.get("ui") or "ar", "expired"))
            return
        job["id"] = uuid.uuid4().hex[:10]
        await begin(context, q, job)
        return

    if act == "tx":
        await send_export(context.bot, q.message.chat_id, context.user_data.get("ui") or "ar", parts[1] if len(parts) > 1 else "")
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
    elif act == "fs":
        # quick start: everything default, only the translation language is asked
        job.update({"fast": True, "src": None, "mode": "sub", "font_i": 0, "size_i": 1,
                    "style_i": 0, "pos_i": 0, "bi": False})
        text, markup = screen_tgt(job)
        await safe_edit(q, text, markup)
    elif act == "tg":
        if parts[2] in TARGETS:
            job["tgt"] = parts[2]
            job["font_i"] = 0
        if job.get("fast"):
            await begin(context, q, job)
            return
        text, markup = screen_mode(job)
        await safe_edit(q, text, markup)
    elif act == "md":
        job["mode"] = "txt" if parts[2] == "txt" else "sub"
        if job["mode"] == "txt":
            await begin(context, q, job)
        else:
            text, markup = screen_settings(job)
            await safe_edit(q, text, markup)
    elif act in ("bi", "bil"):
        text, markup = screen_bi(job, int(parts[2]) if act == "bil" else 0)
        await safe_edit(q, text, markup)
    elif act == "bp":
        v = parts[2]
        job["bi"] = False if v == "off" else "src" if v == "src" else (v if v in LANGS else False)
        text, markup = screen_settings(job)
        await safe_edit(q, text, markup)
        await live_preview(context, q.message.chat_id, job)
    elif act == "bk":
        where = parts[2] if len(parts) > 2 else "set"
        if where == "src":
            job["fast"] = False
        text, markup = {"src": screen_src, "tgt": screen_tgt, "mode": screen_mode}.get(where, screen_settings)(job)
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
        await live_preview(context, q.message.chat_id, job)
    elif act == "sp":
        await live_preview(context, q.message.chat_id, job, force_new=True)
    elif act == "fp":
        group = "ar" if job["tgt"] in ARABIC_SCRIPT else "lat"
        fonts = fonts_for(job["tgt"])
        png = await combo_preview(group, fonts[job["font_i"] % len(fonts)], ui)
        if png:
            await context.bot.send_photo(q.message.chat_id, photo=png, caption=T(ui, "preview_cap"))
    elif act == "go":
        await begin(context, q, job)
    elif act == "qk":
        job["src"] = None
        await begin(context, q, job)


_LIVE_CACHE: dict = {}
LIVE_SAMPLES = {"ar": ("مرحبا، كيف حالك؟", "ar"), "lat": ("Hello, how are you?", "lat")}


async def live_frame(job: dict):
    """One video-like frame showing the subtitle exactly as chosen (font, size, style, position, bilingual)."""
    tgt = job["tgt"]
    group = "ar" if tgt in ARABIC_SCRIPT else "lat"
    fonts = fonts_for(tgt)
    font = fonts[job["font_i"] % len(fonts)]
    bi = job.get("bi")
    sec_group = None
    if bi:
        other = bi if bi not in (True, "src") else (job.get("src") or None)
        if other == tgt:
            sec_group = None
        elif other:
            sec_group = "ar" if other in ARABIC_SCRIPT else "lat"
        else:
            sec_group = "lat" if group == "ar" else "ar"
    key = (group, font, job["size_i"], job["style_i"], job["pos_i"], sec_group)
    if key in _LIVE_CACHE:
        return _LIVE_CACHE[key]
    W, H = 720, 900
    main = LIVE_SAMPLES[group][0]
    mult = SIZES[job["size_i"]][1]
    fs = round(34 * mult * font_scale(font, group))
    style_key = STYLES[job["style_i"] % len(STYLES)]
    an_pos = POSITIONS[job["pos_i"] % len(POSITIONS)][0]
    ycen = {"pos_top": 170, "pos_middle": H // 2, "pos_bottom": H - 190}[an_pos]
    if sec_group:
        ycen -= round(fs * 0.5) if an_pos != "pos_top" else 0
    style = _ass_style_line("M", font, fs, STYLE_DEFS[style_key])
    ev = [f"Dialogue: 1,0:00:00.00,0:00:05.00,M,,0,0,0,,{{\\an5\\pos({W // 2},{ycen})}}{main}"]
    if sec_group:
        sec = LIVE_SAMPLES[sec_group][0]
        sfont = "Noto Naskh Arabic" if sec_group == "ar" else "Noto Sans"
        sfs = round(fs * 0.7 * 1.15 / 0.9)
        ev.append(f"Dialogue: 1,0:00:00.00,0:00:05.00,M,,0,0,0,,{{\\an5\\pos({W // 2},{ycen + round(fs * 1.25)})\\fn{sfont}\\fs{sfs}\\1c&H00D9D9D9&}}{sec}")
    ass = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n" + style
        + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(ev) + "\n"
    )
    workdir = Path(tempfile.mkdtemp(prefix="live_"))
    try:
        (workdir / "p.ass").write_text(ass, encoding="utf-8")
        for bg in (f"gradients=s={W}x{H}:c0=0x2b5876:c1=0xc9a27a:x0=0:y0=0:x1={W}:y1={H}:d=1",
                   f"color=c=0x5a6b7b:s={W}x{H}:d=1"):
            rc, _, err = await run_capture(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", bg,
                                            "-vf", "ass=p.ass", "-frames:v", "1", "p.png"], cwd=str(workdir))
            if rc == 0 and (workdir / "p.png").exists():
                png = (workdir / "p.png").read_bytes()
                if len(_LIVE_CACHE) > 200:
                    _LIVE_CACHE.clear()
                _LIVE_CACHE[key] = png
                return png
        log.warning("live preview failed: %s", err.decode(errors="ignore")[-200:])
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def live_preview(context, chat_id, job, force_new=False):
    """Show/refresh the live preview photo. Once it exists it is edited in place after each change."""
    if not force_new and not job.get("pv"):
        return
    png = await live_frame(job)
    if not png:
        return
    ui = job["ui"]
    mid = job.get("pv")
    if mid and not force_new:
        try:
            await context.bot.edit_message_media(chat_id=chat_id, message_id=mid,
                                                 media=InputMediaPhoto(png, caption=T(ui, "live_cap")))
            return
        except TelegramError as e:
            log.debug("live edit failed: %s", e)
    if mid:
        try:
            await context.bot.delete_message(chat_id, mid)
        except TelegramError:
            pass
    sent = await context.bot.send_photo(chat_id, photo=png, caption=T(ui, "live_cap"))
    job["pv"] = sent.message_id


_COMBO_CACHE: dict = {}


async def combo_preview(group: str, font: str, ui: str):
    """One picture with everything: fonts, sizes, styles, position and the bilingual look (cached)."""
    key = (group, font, ui)
    if key in _COMBO_CACHE:
        return _COMBO_CACHE[key]
    W = 900
    rtl_ui = ui == "ar"
    lab = "Noto Naskh Arabic" if rtl_ui else "Noto Sans"
    sample = "السلام عليكم" if group == "ar" else "Hello world"
    short = "مرحبا" if group == "ar" else "Hello"
    fonts = list(FONTS_AR if group == "ar" else FONTS_LAT)
    ev, styles = [], []
    L = "0:00:00.00,0:00:05.00"

    def rect(x, y, w, h, col="&H000000&", alpha="&H78&", layer=0):
        ev.append(f"Dialogue: {layer},{L},Lbl,,0,0,0,,{{\\an7\\pos(0,0)\\p1\\bord0\\1c{col}\\1a{alpha}}}m {x} {y} l {x + w} {y} l {x + w} {y + h} l {x} {y + h}{{\\p0}}")

    def title(y, text):
        an, x = (9, W - 24) if rtl_ui else (7, 24)
        ev.append(f"Dialogue: 3,{L},Lbl,,0,0,0,,{{\\an{an}\\pos({x},{y})\\fn{lab}\\b1\\fs52\\1c&H0050D0FF&\\bord2}}{text}")

    def small(x, y, text, an=7, fs=40, col="&H00E8E8E8&"):
        ev.append(f"Dialogue: 3,{L},Lbl,,0,0,0,,{{\\an{an}\\pos({x},{y})\\fn{lab}\\b1\\fs{fs}\\1c{col}\\bord2}}{text}")

    for i, key_s in enumerate(STYLES):
        styles.append(_ass_style_line(f"S{i}", font, round(44 * font_scale(font, group)), STYLE_DEFS[key_s]))
    y = 10
    # 1) fonts
    title(y, T(ui, "pv_fonts")); y += 78
    cw, ch = 438, 150
    rect(6, y - 4, W - 12, ch * ((len(fonts) + 1) // 2) + 8)
    for i, name in enumerate(fonts):
        x0, y0 = 12 + (i % 2) * cw, y + (i // 2) * ch
        small(x0 + 10, y0 + 4, f"{i + 1}. {name}")
        ev.append(f"Dialogue: 3,{L},S0,,0,0,0,,{{\\an5\\pos({x0 + cw // 2},{y0 + 104})\\fn{name}\\fs{round(44 * font_scale(name, group))}}}{sample}")
    y += ch * ((len(fonts) + 1) // 2) + 16
    # 2) sizes
    title(y, T(ui, "pv_sizes")); y += 78
    rect(6, y - 4, W - 12, 160)
    for i, (k, mult) in enumerate(SIZES):
        x0 = 12 + i * 292
        small(x0 + 10, y + 4, T(ui, k))
        ev.append(f"Dialogue: 3,{L},S0,,0,0,0,,{{\\an5\\pos({x0 + 146},{y + 108})\\fs{round(32 * mult * font_scale(font, group))}}}{short}")
    y += 176
    # 3) styles
    title(y, T(ui, "pv_styles")); y += 78
    cw, ch = 450, 130
    for i, key_s in enumerate(STYLES):
        cx, cy = (i % 2) * cw, y + (i // 2) * ch
        ev.append(f"Dialogue: 3,{L},S{i},,0,0,0,,{{\\an5\\pos({cx + cw // 2},{cy + 90})}}{sample}")
        small(cx + 14, cy + 4, T(ui, "style_" + key_s))
    y += ch * ((len(STYLES) + 1) // 2) + 14
    # 4) position (three mini screens)
    title(y, T(ui, "pv_pos")); y += 78
    fw, fh = 270, 150
    for i, (k, _) in enumerate(reversed(POSITIONS)):       # top, middle, bottom
        x0 = 15 + i * 295
        rect(x0, y, fw, fh, col="&H303030&", alpha="&H00&")
        yy = {"pos_top": y + 30, "pos_middle": y + fh // 2, "pos_bottom": y + fh - 30}[k]
        ev.append(f"Dialogue: 3,{L},S1,,0,0,0,,{{\\an5\\pos({x0 + fw // 2},{yy})\\fs{round(26 * font_scale(font, group))}}}{short}")
        small(x0 + fw // 2, y + fh + 6, T(ui, k), an=8, fs=40)
    y += fh + 76
    # 5) bilingual
    title(y, "🌐 " + T(ui, "pv_bi")); y += 78
    main, sec = ("مرحبا بكم في البوت", "Welcome to the bot") if group == "ar" else ("Welcome to the bot", "مرحبا بكم في البوت")
    rect(15, y, W - 30, 180, col="&H303030&", alpha="&H00&")
    ev.append(f"Dialogue: 3,{L},S1,,0,0,0,,{{\\an5\\pos({W // 2},{y + 55})\\fs{round(42 * font_scale(font, group))}}}{main}")
    sec_font = "Noto Sans" if group == "ar" else "Noto Naskh Arabic"
    ev.append(f"Dialogue: 3,{L},S1,,0,0,0,,{{\\an5\\pos({W // 2},{y + 125})\\fn{sec_font}\\fs36\\1c&H00D9D9D9&}}{sec}")
    y += 196
    ev.append(f"Dialogue: 3,{L},Lbl,,40,40,0,,{{\\an8\\pos({W // 2},{y})\\fn{lab}\\fs38\\1c&H00F0F0F0&\\bord2}}{T(ui, 'pv_bi_note')}")
    H = y + 150
    ass = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Lbl,Noto Sans,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,7,0,0,0,1\n"
        + "\n".join(styles)
        + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(ev) + "\n"
    )
    workdir = Path(tempfile.mkdtemp(prefix="combo_"))
    try:
        (workdir / "p.ass").write_text(ass, encoding="utf-8")
        for bg in (f"gradients=s={W}x{H}:c0=0x2b5876:c1=0xc9a27a:x0=0:y0=0:x1={W}:y1={H}:d=1",
                   f"color=c=0x5a6b7b:s={W}x{H}:d=1"):
            rc, _, err = await run_capture(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", bg,
                                            "-vf", "ass=p.ass", "-frames:v", "1", "p.png"], cwd=str(workdir))
            if rc == 0 and (workdir / "p.png").exists():
                _COMBO_CACHE[key] = (workdir / "p.png").read_bytes()
                return _COMBO_CACHE[key]
        log.warning("combo preview failed: %s", err.decode(errors="ignore")[-200:])
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


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
    cols, cw, ch = 2, 450, 130
    rows = (len(STYLES) + cols - 1) // cols
    width, height = cols * cw, rows * ch
    size = round(54 * font_scale(font, group))
    label_font = "Noto Naskh Arabic" if ui == "ar" else "Noto Sans"
    styles, events = [], []
    for i, key_s in enumerate(STYLES):
        cx, cy = (i % cols) * cw, (i // cols) * ch
        styles.append(_ass_style_line(f"S{i}", font, size, STYLE_DEFS[key_s]))
        events.append(f"Dialogue: 1,0:00:00.00,0:00:05.00,S{i},,0,0,0,,{{\\an5\\pos({cx + cw // 2},{cy + ch // 2 + 22})}}{sample}")
        events.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,Lbl,,0,0,0,,{{\\an7\\pos({cx + 12},{cy + 6})\\fn{label_font}\\b1\\fs28}}{T(ui, 'style_' + key_s)}")
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
    sample = "السلام عليكم ١٢٣" if group == "ar" else "Hello world 123"
    row_h, width = 130, 720
    height = row_h * len(fonts) + 16
    events = []
    for i, name in enumerate(fonts):
        y0 = 8 + row_h * i
        size = round(56 * font_scale(name, group))
        if i:   # thin separator between rows
            events.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,L,,0,0,0,,{{\\an7\\pos(0,{y0})\\p1\\bord0\\1c&H00444444&}}m 0 0 l {width} 0 l {width} 2 l 0 2{{\\p0}}")
        events.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,L,,0,0,0,,{{\\an7\\pos(20,{y0 + 8})\\fnNoto Sans\\b1\\fs32\\1c&H0050D0FF&}}{i + 1}. {name}")
        events.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,S,,0,0,0,,{{\\an5\\pos({width // 2},{y0 + 84})\\fn{name}\\fs{size}}}{sample}")
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


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        return
    await update.effective_message.reply_text(await asyncio.to_thread(stats_text))


async def cmd_allow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/allow 123456  - add a person (the bot then only serves the list) ;  /allow  - show the list ;  /disallow 123456"""
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        return
    msg = update.effective_message
    cmd = (msg.text or "").split()[0].lstrip("/").split("@")[0].lower()
    ids = [int(x) for x in re.findall(r"\d+", " ".join((msg.text or "").split()[1:]))]
    if not ids:
        pool = sorted(ALLOWED_ENV | ALLOWED_DB)
        await msg.reply_text("المسموح لهم: " + (", ".join(map(str, pool)) if pool else "الجميع (القائمة فارغة)")
                             + "\n\nللإضافة: /allow 123456\nللحذف: /disallow 123456")
        return
    for i in ids:
        if cmd == "disallow":
            ALLOWED_DB.discard(i)
        else:
            ALLOWED_DB.add(i)

    def run():
        if DB_OK:
            with _db() as c:
                for i in ids:
                    if cmd == "disallow":
                        c.execute("delete from allowed where uid = %s", (i,))
                    else:
                        c.execute("insert into allowed (uid) values (%s) on conflict do nothing", (i,))
    try:
        await asyncio.to_thread(run)
        note = "" if DB_OK else "\n⚠️ قاعدة البيانات غير متصلة: التغيير يضيع عند إعادة التشغيل."
        await msg.reply_text(("✅ تمت الإضافة" if cmd != "disallow" else "🗑 تم الحذف") + f": {', '.join(map(str, ids))}\n"
                             "تذكير: إذا صارت القائمة غير فارغة فالبوت يخدم المسموح لهم فقط." + note)
    except Exception as e:
        await msg.reply_text(f"❌ تعذّر الحفظ: {str(e)[:150]}")


async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin only: the last log lines as a file."""
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        return
    text = "\n".join(RING.lines) or "(empty)"
    await update.effective_message.reply_document(document=text.encode("utf-8"), filename="subvid-log.txt")


async def cmd_localapi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin only: /localapi <api_id> <api_hash> saves the keys of my.telegram.org for the own Bot API server."""
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        return
    msg = update.effective_message
    args = context.args or []
    ok = len(args) == 2 and args[0].isdigit() and re.fullmatch(r"[0-9a-fA-F]{32}", args[1])
    try:
        await msg.delete()   # keep the hash out of the chat history
    except TelegramError:
        pass
    if not ok:
        got = " ".join(args)
        n_id = len(args[0]) if args else 0
        n_h = len(args[1]) if len(args) > 1 else 0
        hint = (f"عدد الأجزاء: {len(args)} (المطلوب 2)\nطول الـapi_id: {n_id}، هل كله أرقام: {bool(args) and args[0].isdigit()}\n"
                f"طول الـapi_hash: {n_h} (المطلوب 32)، هل أحرف وأرقام إنجليزية فقط: {n_h > 0 and bool(re.fullmatch(r'[0-9a-fA-F]+', args[1]))}")
        await context.bot.send_message(msg.chat_id, "الصيغة: /localapi api_id api_hash\n(مسافة واحدة بينهما، بدون أقواس)\n\n" + hint)
        return
    (BASE_DIR / "localapi.json").write_text(json.dumps({"api_id": args[0], "api_hash": args[1]}), encoding="utf-8")
    state = "مفعّل الآن" if LOCAL else "سيتفعّل بعد إعادة بناء السيرفر (تلقائيًا في الليل)"
    await context.bot.send_message(msg.chat_id, f"✅ تم حفظ المفاتيح، وحذفت رسالتك. السيرفر المحلي: {state}.")


async def cmd_setgroq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin only: /setgroq <key> saves a new Groq key, deletes the message, tests the key and uses it at once."""
    global GROQ_API_KEY
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        return
    msg = update.effective_message
    key = "".join((context.args or [""])[0].split()).strip("'\"")
    try:
        await msg.delete()
    except TelegramError:
        pass
    if not key.startswith("gsk_") or len(key) < 30:
        await context.bot.send_message(msg.chat_id, "الصيغة: /setgroq gsk_xxxxx (المفتاح يبدأ بـ gsk_)")
        return

    def _test():
        return httpx.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=20).status_code

    try:
        code = await asyncio.to_thread(_test)
    except Exception as e:
        await context.bot.send_message(msg.chat_id, f"⚠️ ما قدرت أتحقق من المفتاح ({type(e).__name__}). لم يُحفظ.")
        return
    if code != 200:
        await context.bot.send_message(msg.chat_id, f"❌ المفتاح مرفوض من Groq (HTTP {code}). انسخه من جديد من console.groq.com. لم يُحفظ.")
        return
    try:
        d = json.loads((BASE_DIR / "secrets.json").read_text(encoding="utf-8"))
    except Exception:
        d = {}
    d["GROQ_API_KEY"] = key
    (BASE_DIR / "secrets.json").write_text(json.dumps(d), encoding="utf-8")
    GROQ_API_KEY = key
    await context.bot.send_message(msg.chat_id, "✅ المفتاح صحيح، حُفظ وبدأ استخدامه الآن، وحذفت رسالتك.")


LAST_DEBUG: dict = {}


def speech_regions(path: str):
    """Where the audio really contains speech (Silero VAD that ships with faster-whisper): [(start, end)] seconds."""
    try:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        audio = decode_audio(path, sampling_rate=16000)
        ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300))
        return [(x["start"] / 16000, x["end"] / 16000) for x in ts]
    except Exception as e:
        log.warning("speech detection failed: %s", str(e)[:150])
        return []


def _tm(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def build_debug_report(d: dict) -> str:
    L = [f"SubVid debug report - {d['lang']} -> {d['tgt']}, {_tm(d['duration'])} ({d['size'][0]}x{d['size'][1]}), "
         f"delay={d['delay']}s, chars/line={d['max_chars']}", ""]
    # 1. raw transcript with word times
    L.append("=== 1. RAW TRANSCRIPT (what the speech model heard) ===")
    for a, b, t in d["raw"]:
        L.append(f"[{_tm(a)}-{_tm(b)}] {t}")
        ws = d["words"].get((a, b)) or []
        if ws:
            L.append("    " + " ".join(f"{w[2].strip()}@{w[0]:.2f}" for w in ws))
    # 2. second model
    L += ["", "=== 2. SECOND MODEL (other Whisper version) - words that differ ==="]
    norm = lambda x: re.findall(r"\w+", _norm_h(x))
    main_words = [w for _, _, t in d["raw"] for w in norm(t)]
    alt_words = norm(d["alt"]) if d["alt"] else []
    if alt_words:
        ms, as_ = set(main_words), set(alt_words)
        L.append("only in the second model (maybe missed by the first): " + (" ".join(sorted(as_ - ms)) or "-"))
        L.append("only in the first model: " + (" ".join(sorted(ms - as_)) or "-"))
    else:
        L.append("(no second opinion available)")
    # 3. speech without transcript
    L += ["", "=== 3. SPEECH DETECTED IN THE AUDIO BUT NOT TRANSCRIBED ==="]
    spans = [(ws[0][0], ws[-1][1]) for ws in d["words"].values() if ws] or [(a, b) for a, b, _ in d["raw"]]
    gaps = []
    for a, b in d["vad"]:
        t = a
        for x, y in sorted(spans):
            if y <= t:
                continue
            if x >= b:
                break
            if x - t > 0.5:
                gaps.append((t, min(x, b)))
            t = max(t, y)
        if b - t > 0.5:
            gaps.append((t, b))
    L += [f"{_tm(a)}-{_tm(b)} ({b - a:.1f}s)" for a, b in gaps] or ["none (or speech detection unavailable)" if not d["vad"] else "none"]
    # 4. translation per segment
    L += ["", "=== 4. TRANSLATION PER SEGMENT (! = suspiciously short / missing words) ==="]
    for (a, b, src), (_, _, tr) in zip(d["src"], d["final"]):
        ratio = len(tr.split()) / max(1, len(src.split()))
        L.append(f"[{_tm(a)}] {'!' if ratio < 0.55 else ' '} {src}\n          -> {tr}")
    if len(d["src"]) != len(d["final"]):
        L.append(f"(note: {len(d['src'])} source lines but {len(d['final'])} translated lines)")
    # 5. cues
    L += ["", "=== 5. FINAL SUBTITLES (shift = how much later than planned) ==="]
    for (a, b, t), (pa, _, _) in zip(d["cues"], d["planned"]):
        sh = a - pa
        L.append(f"[{a:6.2f}-{b:6.2f}] {'shift %+.2fs ' % sh if abs(sh) >= 0.1 else ''}{t.replace(chr(1), '').replace(chr(10), ' / ')}")
    return "\n".join(L)


async def cmd_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin only: report about the last video you sent yourself (what was heard, translated, and when it is shown)."""
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        return
    if not LAST_DEBUG:
        await update.effective_message.reply_text("ما فيه تقرير بعد. أرسل فيديو بنفسك أولًا ثم أرسل /debug.")
        return
    text = await asyncio.to_thread(build_debug_report, dict(LAST_DEBUG))
    await update.effective_message.reply_document(document=text.encode("utf-8"), filename="subvid-debug.txt")


async def cmd_delay(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin only: /delay 0.15 shifts all subtitles later (negative = earlier), in seconds. Applies to the next videos."""
    global SUB_DELAY
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        return
    msg = update.effective_message
    if not context.args:
        await msg.reply_text(f"التأخير الحالي: {SUB_DELAY:+.2f} ثانية.\nلتغييره: /delay 0.15 (موجب = الترجمة تتأخر، سالب = تتقدم)")
        return
    try:
        v = float(context.args[0].replace(",", "."))
    except ValueError:
        await msg.reply_text("الصيغة: /delay 0.15")
        return
    v = max(-1.0, min(1.0, v))
    try:
        d = json.loads((BASE_DIR / "secrets.json").read_text(encoding="utf-8"))
    except Exception:
        d = {}
    d["SUB_DELAY"] = v
    (BASE_DIR / "secrets.json").write_text(json.dumps(d), encoding="utf-8")
    SUB_DELAY = v
    await msg.reply_text(f"✅ التأخير صار {v:+.2f} ثانية للفيديوهات القادمة.")


BUILD_SCRIPT = r"""
set -ex
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends make git zlib1g-dev libssl-dev gperf cmake g++ ca-certificates
rm -rf /tmp/tgsrc /tmp/tgout
git clone --recursive --depth 1 --shallow-submodules https://github.com/tdlib/telegram-bot-api.git /tmp/tgsrc
mkdir /tmp/tgsrc/build && cd /tmp/tgsrc/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=/tmp/tgout ..
cmake --build . --target install -j4
mkdir -p /app/bin
cp /tmp/tgout/bin/telegram-bot-api /app/bin/telegram-bot-api
strip /app/bin/telegram-bot-api
rm -rf /tmp/tgsrc /tmp/tgout
"""


async def cmd_buildlocal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin only: builds the own Bot API server inside the running container (10-20 min), then reports."""
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        return
    msg = update.effective_message
    if TGAPI_BIN:
        await msg.reply_text("✅ السيرفر المحلي مبني أصلًا. أرسل /update ليبدأ التحويل.")
        return
    if context.application.bot_data.get("building"):
        await msg.reply_text("⏳ البناء شغّال الآن.")
        return
    context.application.bot_data["building"] = True
    await msg.reply_text("🔧 بدأ بناء السيرفر المحلي (10 إلى 20 دقيقة). سأخبرك عند الانتهاء. البوت يشتغل طبيعي أثناء ذلك.")
    chat_id = msg.chat_id

    async def _run():
        try:
            proc = await asyncio.create_subprocess_exec("bash", "-c", BUILD_SCRIPT, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.STDOUT)
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=3600)
            tail = out.decode("utf-8", "replace")[-1200:]
            if proc.returncode == 0 and (BASE_DIR / "bin" / "telegram-bot-api").exists():
                await context.bot.send_message(chat_id, "✅ انبنى السيرفر المحلي. الآن أرسل /update ليبدأ التحويل (يتوقف البوت حوالي 11 دقيقة مرة وحدة).")
            else:
                await context.bot.send_message(chat_id, f"❌ فشل البناء (رمز {proc.returncode}). آخر الأسطر:\n{tail}")
        except Exception as e:
            await context.bot.send_message(chat_id, f"❌ فشل البناء: {type(e).__name__}: {e}")
        finally:
            context.application.bot_data["building"] = False

    asyncio.create_task(_run())


async def cmd_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin only: restart the bot; on the own server the container pulls the newest code from GitHub on start."""
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        return
    if ACTIVE:
        await update.effective_message.reply_text("⏳ فيه معالجة شغّالة الآن. أعد /update بعد ما تنتهي.")
        return
    await update.effective_message.reply_text("🔄 جاري التحديث... يرجع البوت خلال نحو 30 ثانية.")
    asyncio.get_running_loop().call_later(1.5, lambda: os._exit(0))


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
        f"الخطوط: {', '.join(FONTS_AR)}\n"
        f"قاعدة البيانات: {DB_STATUS}"
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


_BG_TASKS: list = []


async def post_shutdown(app: Application):
    for t in _BG_TASKS:
        t.cancel()


async def db_keepalive(bot):
    """Supabase's free plan pauses an idle project after a week: touch it every day."""
    while True:
        await asyncio.sleep(24 * 3600)
        try:
            def ping():
                with _db() as c:
                    c.execute("select 1").fetchone()
            await asyncio.to_thread(ping)
        except Exception as e:
            log.warning("database keep-alive failed: %s", str(e)[:150])
            await notify_admin(bot, f"⚠️ تعذّر الاتصال بقاعدة البيانات: {str(e)[:200]}", key="db-keepalive")


async def post_init(app: Application):
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            log.critical("%s not found in PATH - deploy with the Dockerfile!", tool)
    await filter_fonts()
    if GROQ_API_KEY:
        log.info("Groq enabled: ASR=%s LLM=%s (local Whisper loads only as a fallback)", GROQ_ASR_MODEL, GROQ_LLM_MODEL)
    else:
        await asyncio.to_thread(get_model)  # warm up so the first request is fast
    if DB_OK:
        _BG_TASKS.append(asyncio.get_running_loop().create_task(db_keepalive(app.bot)))
    log.info("startup complete, memory in use: %s", mem_now())
    await notify_admin(app.bot, f"🟢 البوت اشتغل (تشغيل جديد)\nالذاكرة: {mem_now()}\nقاعدة البيانات: {DB_STATUS}", key="startup")


# -------------------------------------------------------------------- main
def _port_open(port: int) -> bool:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def start_local_api() -> bool:
    """Starts our own telegram-bot-api server. On the first start it moves the bot from Telegram's cloud server to it."""
    import subprocess
    api_id, api_hash = LOCAL_API
    work = Path(tempfile.gettempdir()) / "tgapi"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    logf = open(work.parent / "tgapi.log", "ab")
    proc = subprocess.Popen([TGAPI_BIN, "--local", f"--api-id={api_id}", f"--api-hash={api_hash}",
                             "--http-port=8081", f"--dir={work}", f"--temp-dir={work}/tmp"],
                            stdout=logf, stderr=logf)
    for _ in range(60):
        if proc.poll() is not None:
            log.error("telegram-bot-api exited with code %s", proc.returncode)
            return False
        if _port_open(8081):
            break
        time.sleep(1)
    else:
        proc.kill()
        return False
    flag = BASE_DIR / ".switched_to_local"
    if not flag.exists():   # the bot may only live on one server: log out of the cloud one first (Telegram then asks to wait 10 min)
        base = f"https://api.telegram.org/bot{TOKEN}"
        try:
            if ADMIN_ID:
                httpx.post(f"{base}/sendMessage", data={"chat_id": ADMIN_ID, "text": "SubVid: switching to the own Bot API server. The bot will be offline for about 11 minutes, once."}, timeout=20)
            r = httpx.post(f"{base}/logOut", timeout=30)
            log.info("cloud logOut: %s", r.text[:100])
        except Exception as e:
            log.warning("cloud logOut failed: %s", e)
        flag.write_text("1")
        log.info("waiting 11 minutes before logging in on the own server")
        time.sleep(660)

    def _watch():   # if the local server ever dies, restart the whole container so it comes back
        proc.wait()
        log.error("telegram-bot-api stopped, restarting")
        os._exit(1)
    threading.Thread(target=_watch, daemon=True).start()
    return True


def main():
    global LOCAL, TG_UPLOAD_LIMIT, UPLOAD_MB
    if LOCAL and not start_local_api():
        log.error("own Bot API server failed to start; using Telegram's cloud server")
        LOCAL = False
        TG_UPLOAD_LIMIT = 49 * 1024 * 1024
        UPLOAD_MB = 44
    builder = ApplicationBuilder().token(TOKEN)
    if LOCAL:
        builder = builder.base_url("http://127.0.0.1:8081/bot").base_file_url("http://127.0.0.1:8081/file/bot").local_mode(True)
    app = (
        builder
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .concurrent_updates(True)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(300)  # video upload
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("language", cmd_language))
    app.add_handler(CommandHandler("fonttest", cmd_fonttest))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("diag", cmd_diag))
    app.add_handler(CommandHandler("logs", cmd_logs))
    app.add_handler(CommandHandler("update", cmd_update))
    app.add_handler(CommandHandler("localapi", cmd_localapi))
    app.add_handler(CommandHandler("buildlocal", cmd_buildlocal))
    app.add_handler(CommandHandler("setgroq", cmd_setgroq))
    app.add_handler(CommandHandler("debug", cmd_debug))
    app.add_handler(CommandHandler("delay", cmd_delay))
    app.add_handler(CommandHandler(["stats", "status"], cmd_stats))
    app.add_handler(CommandHandler(["allow", "disallow"], cmd_allow))
    app.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE
            & ((filters.TEXT & ~filters.COMMAND) | filters.VIDEO | filters.Document.VIDEO
               | filters.Document.FileExtension("txt") | filters.Document.FileExtension("srt")),
            on_input,
        )
    )
    app.add_handler(
        CallbackQueryHandler(on_button, pattern=r"^(ui|src|sl|ss|tgl|tg|md|cy|fp|sp|go|qk|fs|tx|rt|bi|bil|bp|bk|x)(\||$)")
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
    else:  # no public URL (own server / local): long polling, nothing needs to be exposed
        load_prefs()
        app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
