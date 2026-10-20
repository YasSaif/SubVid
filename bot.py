"""SubVid Ai - Telegram bot: burn subtitles into a video or extract them as text.

Flow: pick UI language -> send video/link -> source language (auto/manual)
      -> target language -> output type -> (style settings) -> process.
Pipeline: download -> audio (ffmpeg) -> transcribe (faster-whisper)
          -> translate (Google API / gtx / MyMemory) -> clean + wrap -> burn or send text.
"""
import asyncio
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
from collections import OrderedDict
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
MAX_MINUTES = int(os.environ.get("MAX_MINUTES", "3"))
COOKIES_FILE = os.environ.get("YTDLP_COOKIES")  # optional cookies.txt for yt-dlp
SS = max(1, min(3, int(os.environ.get("SUB_SUPERSAMPLE", "2"))))  # render subtitles at SSx size, then shrink (avoids glyph boxes at tiny sizes)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # optional: far more accurate transcription + translation
GROQ_ASR_MODEL = os.environ.get("GROQ_ASR_MODEL", "whisper-large-v3")  # most accurate; turbo is used as fallback
GROQ_ASR_FALLBACK = "whisper-large-v3-turbo"
SUB_DELAY = float(os.environ.get("SUB_DELAY", "0.12"))  # seconds subtitles are shifted later (text was appearing before the voice)
GROQ_PROMPT = os.environ.get("GROQ_PROMPT", "")  # optional: names/terms that appear in your clips
GROQ_LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
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
STYLES = ["classic", "box", "yellow"]
STYLE_DEFS = {  # ASS colours are &HAABBGGRR (AA=00 opaque, FF transparent)
    "classic": "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0",
    "yellow": "PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0",
    "box": "PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BackColour=&H80000000,BorderStyle=3,Outline=3,Shadow=0",
}
POSITIONS = [("pos_bottom", 2), ("pos_middle", 5), ("pos_top", 8)]


def fonts_for(tgt: str):
    return FONTS_AR if tgt in ARABIC_SCRIPT else FONTS_LAT


# ---------------------------------------------------------------- strings
STR = {
    "ar": {
        "pick_ui": "🌐 اختر لغة الواجهة\nChoose the interface language",
        "welcome": "أهلاً بك في SubVid Ai 🎬\n\nأرسل رابط فيديو (يوتيوب، تيك توك، رابط مباشر) أو ملف فيديو.\nلتغيير لغة الواجهة: /language",
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
        "style_classic": "كلاسيكي", "style_box": "خلفية", "style_yellow": "أصفر",
        "pos_bottom": "أسفل", "pos_middle": "وسط", "pos_top": "أعلى",
        "start": "▶️ ابدأ", "cancel": "إلغاء", "cancelled": "تم الإلغاء.",
        "expired": "انتهت صلاحية الطلب، أعد إرسال الفيديو.",
        "queued": "⏳ جاري التجهيز...",
        "queue": "⏳ ترتيبك في الطابور: {n} قبلك.",
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
        "too_long": "الحد الأقصى لمدة المقطع {m} دقائق.",
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
        "style_classic": "Classic", "style_box": "Box", "style_yellow": "Yellow",
        "pos_bottom": "Bottom", "pos_middle": "Middle", "pos_top": "Top",
        "start": "▶️ Start", "cancel": "Cancel", "cancelled": "Cancelled.",
        "expired": "This request expired, please send the video again.",
        "queued": "⏳ Preparing...",
        "queue": "⏳ Your place in the queue: {n} ahead of you.",
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
        "too_long": "Maximum clip length is {m} minutes.",
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


def transcribe_groq(audio_path: str, language=None):
    last = None
    for model in (GROQ_ASR_MODEL, GROQ_ASR_FALLBACK):
        try:
            with open(audio_path, "rb") as f:
                data = {"model": model, "response_format": "verbose_json", "temperature": "0",
                        "timestamp_granularities[]": ["word", "segment"]}
                if language:
                    data["language"] = language
                if GROQ_PROMPT:
                    data["prompt"] = GROQ_PROMPT[:800]
                r = httpx.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    data=data, files={"file": ("audio.flac", f, "audio/flac")}, timeout=120,
                )
            r.raise_for_status()
            break
        except Exception as e:
            last = e
            log.warning("Groq ASR %s failed: %s", model, str(e)[:120])
    else:
        raise last
    d = r.json()
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
    return out, lang


def transcribe(audio_path: str, language=None):
    """language=None -> auto-detect; otherwise skip detection (faster)."""
    if GROQ_API_KEY:
        try:
            return transcribe_groq(audio_path, language)
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
    return out, info.language


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


def _llm_batch(texts, src_name, tgt_name, fix_only):
    task = (
        f"These are consecutive subtitle lines from a video, auto-transcribed from speech ({src_name}). "
        "The transcription may contain mistakes (wrong or misheard words, names, missing letters). "
        "NEVER delete or skip words or sentences: every spoken word must stay in the output. "
        "If a word looks garbled, replace it with the most likely intended word given the context. "
    )
    if fix_only:
        task += "Fix obvious transcription mistakes using the context. Keep the same language and meaning."
    else:
        task += (f"First fix obvious mistakes using the context, then translate each line into natural, "
                 f"fluent {tgt_name}, as a native speaker would say it. Keep proper names correct.")
    task += (' Return JSON {"t": [...]} with EXACTLY the same number of items, in the same order, '
             "one output string per input line. No explanations.")
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_LLM_MODEL, "temperature": 0.2, "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": task},
                           {"role": "user", "content": json.dumps({"lines": texts}, ensure_ascii=False)}]},
        timeout=60,
    )
    r.raise_for_status()
    out = json.loads(r.json()["choices"][0]["message"]["content"])["t"]
    if not isinstance(out, list) or len(out) != len(texts):
        raise ValueError("line count mismatch")
    return [str(x).strip() for x in out]


def translate_segments_llm(segments, src: str, tgt: str):
    """Context-aware translation + correction via Groq (Llama). Raises on failure."""
    names = {c: n for n, c in LANG_NAME_TO_CODE.items()}
    src_name, tgt_name = names.get(src, src), names.get(tgt, tgt)
    texts = [" ".join(t.split()) for _, _, t in segments]
    result = []
    for i in range(0, len(texts), 30):
        batch = texts[i:i + 30]
        for attempt in range(2):
            try:
                result += _llm_batch(batch, src_name, tgt_name, fix_only=(src == tgt))
                break
            except Exception as e:
                if attempt == 1:
                    raise
                log.warning("LLM batch failed, retrying: %s", str(e)[:120])
                time.sleep(2)
    return [(s, e, r or t) for (s, e, _), r, t in zip(segments, result, texts)]


def translate_segments(segments, src: str, tgt: str):
    """Translate in a few small batches (one segment per line)."""
    if GROQ_API_KEY:
        try:
            return translate_segments_llm(segments, src, tgt)
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


def make_cues(segments, max_chars: int):
    """Split long segments into several timed cues of at most two short lines."""
    cues = []
    for start, end, text in segments:
        words = text.split()
        if not words:
            continue
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
        t, dur = start, end - start
        for c in chunks:
            line = " ".join(c)
            d = dur * len(line) / total
            cues.append((t, t + d, wrap2(line, max_chars)))
            t += d
    return cues


# -------------------------------------------------------------------- srt
def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(cues) -> str:
    blocks = []
    for i, (start, end, text) in enumerate(cues, start=1):
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
            "stream=width,height:stream_tags=rotate:stream_side_data=rotation:format=duration",
            "-of", "json", str(video),
        ],
        timeout=60,
    )
    info = json.loads(raw or b"{}")
    stream = (info.get("streams") or [{}])[0]
    w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
    rot = (stream.get("tags") or {}).get("rotate")
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    if rot is not None and abs(int(float(rot))) % 180 == 90:
        w, h = h, w
    duration = float((info.get("format") or {}).get("duration") or 0)
    return w, h, duration


async def burn_subtitles(video: Path, srt_name: str, workdir: Path, w: int, h: int, job: dict) -> Path:
    scale = ""
    short_side = min(w, h)
    if short_side and short_side != MAX_RES:  # fixed output size: small sources are upscaled so the text stays sharp
        f = MAX_RES / short_side
        w, h = int(w * f) // 2 * 2, int(h * f) // 2 * 2
        scale = f"scale={w}:{h},"
    portrait = h > w

    fonts = fonts_for(job["tgt"])
    font = fonts[job["font_i"] % len(fonts)]
    log.info("burn: font=%s size=x%s style=%s pos=%s res=%sx%s", font, SIZES[job["size_i"]][1],
             STYLES[job["style_i"]], POSITIONS[job["pos_i"]][0], w, h)
    mult = SIZES[job["size_i"]][1]
    base = 12 if portrait else 17  # relative to a 288px-high canvas
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
            lines[2] = "{\\an%d}" % align + lines[2]
        tagged.append("\n".join(lines))
    srt_name = "subs_burn.srt"
    (workdir / srt_name).write_text("\n\n".join(tagged) + "\n", encoding="utf-8")
    out = workdir / "output.mp4"
    # srt is referenced by relative name and ffmpeg runs inside workdir -> no path escaping issues
    w, h = w // 2 * 2, h // 2 * 2
    if SS > 1:
        vf = (f"{scale}scale={w * SS}:{h * SS}:flags=bicubic,subtitles={srt_name}:force_style='{style}',"
              f"scale={w}:{h}:flags=area")
    else:
        vf = f"{scale}subtitles={srt_name}:force_style='{style}'"
    await run_cmd(
        [
            "ffmpeg", "-y", "-v", "verbose", "-threads", "1", "-filter_threads", "1", "-i", str(video), "-vf", vf,
            "-c:v", "libx264", "-preset", "superfast", "-crf", "28", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-threads", "1", "-movflags", "+faststart", str(out),
        ],
        cwd=str(workdir),
        timeout=1500,
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


async def process(bot, chat_id: int, job: dict, status):
    ui, mode, tgt = job["ui"], job["mode"], job["tgt"]
    workdir = Path(tempfile.mkdtemp(prefix="subvid_"))
    st = Stages()
    PENDING.append(job["id"])
    try:
        ahead = PENDING.index(job["id"])
        if ahead > 0:
            await set_status(status, T(ui, "queue", n=ahead))
        async with JOB_SEM:
            st.reset()  # don't count the time spent waiting in the queue
            await set_status(status, T(ui, "p_dl"))
            video = await fetch_input(bot, job["kind"], job["data"], workdir, audio_only=(mode == "txt"))
            st.mark("download")

            w, h, duration = await probe(video)
            st.mark("probe")
            if duration and duration > MAX_MINUTES * 60:
                raise UserError("too_long", m=MAX_MINUTES)

            await set_status(status, T(ui, "p_audio"))
            # clean the audio (rumble/hiss, uneven volume) so recognition doesn't depend on recording quality
            audio = workdir / "audio.flac"
            await run_cmd(
                ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                 "-af", "highpass=f=80,lowpass=f=7600,afftdn=nf=-28,dynaudnorm=f=150:g=15",
                 str(audio)]
            )
            st.mark("audio")

            await set_status(status, T(ui, "p_tr" if job["src"] else "p_tr_auto"))
            segments, lang = await asyncio.to_thread(transcribe, str(audio), job["src"])
            st.mark("transcribe")
            log.info("raw transcript (%s): %s", lang, " | ".join(t for _, _, t in segments)[:600])
            if not segments:
                raise UserError("no_speech")

            if lang != tgt or GROQ_API_KEY:
                await set_status(status, T(ui, "p_trans"))
                segments = await asyncio.to_thread(translate_segments, segments, lang, tgt)
                st.mark("translate")

            keep_joiners = tgt in ("fa", "ur")
            segments = [(s, e, clean_text(t, keep_joiners)) for s, e, t in segments]
            if tgt in ARABIC_SCRIPT:
                raw_odd = odd_chars(" ".join(t for _, _, t in segments))
                segments = [(s, e, normalize_ar(t)) for s, e, t in segments]
                log.info("subtitle text: odd chars before=[%s] after=[%s]", raw_odd,
                         odd_chars(" ".join(t for _, _, t in segments)))
            segments = [x for x in segments if x[2]]
            if not segments:
                raise UserError("no_speech")

            portrait = bool(w and h and h > w)
            mult = SIZES[job["size_i"]][1]
            base_chars = 28 if portrait else 40
            max_chars = max(14, int(base_chars / mult)) if mode == "sub" else 40
            cues = make_cues(segments, max_chars)
            srt_text = build_srt(cues)
            (workdir / "subs.srt").write_text(srt_text, encoding="utf-8")
            plain = "\n".join(t for _, _, t in segments)

            if mode == "txt":
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
            else:
                await set_status(status, T(ui, "p_burn"))
                out = await burn_subtitles(video, "subs.srt", workdir, w, h, job)
                st.mark("burn")
                if out.stat().st_size > TG_UPLOAD_LIMIT:
                    with open(workdir / "subs.srt", "rb") as f:
                        await bot.send_document(chat_id, f, filename="subtitles.srt",
                                                caption=T(ui, "too_big"))
                else:
                    await set_status(status, T(ui, "p_up"))
                    with open(out, "rb") as f:
                        await bot.send_video(chat_id, f, supports_streaming=True,
                                             caption=T(ui, "caption_burn"))
                st.mark("upload")
        try:
            await status.delete()
        except TelegramError:
            pass
    except UserError as e:
        await set_status(status, "⚠️ " + T(ui, e.key, **e.kw))
    except Exception:
        log.exception("job failed")
        await set_status(status, "❌ " + T(ui, "generic"))
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
        [btn(f"{T(ui, 'style')}: {T(ui, 'style_' + STYLES[job['style_i']])}", f"cy|{key}|style")],
        [btn(f"{T(ui, 'pos')}: {T(ui, POSITIONS[job['pos_i']][0])}", f"cy|{key}|pos")],
        [btn(T(ui, "start"), f"go|{key}")],
        [btn(T(ui, "cancel"), f"x|{key}")],
    ]
    return T(ui, "settings"), InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------- handlers
def new_job(kind: str, data: str, ui: str) -> dict:
    key = uuid.uuid4().hex[:10]
    JOBS[key] = {
        "id": key, "kind": kind, "data": data, "ui": ui,
        "src": None, "tgt": ui if ui in TARGETS else "ar", "mode": "sub",
        "font_i": 0, "size_i": 1, "style_i": 0, "pos_i": 0,
    }
    while len(JOBS) > 300:
        JOBS.popitem(last=False)
    return JOBS[key]


async def safe_edit(q, text, markup=None):
    try:
        await q.edit_message_text(text, reply_markup=markup)
    except TelegramError as e:  # e.g. "message is not modified"
        log.debug("edit skipped: %s", e)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    ui = context.user_data.get("ui")
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

    ui = context.user_data.get("ui")
    if not kind:
        await msg.reply_text(T(ui or "ar", "send_media"))
        return
    if not ui:  # ask for the interface language first, keep the media for afterwards
        context.user_data["pending"] = (kind, data)
        text, markup = screen_ui_picker()
        await msg.reply_text(text, reply_markup=markup)
        return

    job = new_job(kind, data, ui)
    text, markup = screen_src(job)
    await msg.reply_text(text, reply_markup=markup)


async def begin(context, q, job):
    JOBS.pop(job["id"], None)
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
        pending = context.user_data.pop("pending", None)
        if pending:
            job = new_job(pending[0], pending[1], ui)
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
        text, markup = screen_settings(job)
        await safe_edit(q, text, markup)
    elif act == "fp":
        group = "ar" if job["tgt"] in ARABIC_SCRIPT else "lat"
        png = await font_preview(group)
        if png:
            await context.bot.send_photo(q.message.chat_id, photo=png, caption=T(ui, "font_preview_cap"))
    elif act == "go":
        await begin(context, q, job)


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
    app.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE
            & ((filters.TEXT & ~filters.COMMAND) | filters.VIDEO | filters.Document.VIDEO),
            on_input,
        )
    )
    app.add_handler(
        CallbackQueryHandler(on_button, pattern=r"^(ui|src|sl|ss|tgl|tg|md|cy|fp|go|x)(\||$)")
    )
    app.add_error_handler(on_error)

    if WEBHOOK_URL:
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