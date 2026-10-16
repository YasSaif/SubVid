"""SubVid Ai - Telegram bot: burn Arabic subtitles into a video or extract them as text.

Pipeline: download -> extract audio (ffmpeg) -> transcribe (faster-whisper)
          -> translate to Arabic (Google via deep-translator) -> burn / send text.
"""
import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
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
MAX_MINUTES = int(os.environ.get("MAX_MINUTES", "10"))
COOKIES_FILE = os.environ.get("YTDLP_COOKIES")  # optional cookies.txt for yt-dlp
TG_UPLOAD_LIMIT = 49 * 1024 * 1024  # Bot API upload limit is 50 MB

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("subvid")

URL_RE = re.compile(r"https?://\S+")
JOB_SEM = asyncio.Semaphore(1)  # one heavy job at a time (protects RAM/CPU)
JOBS: "OrderedDict[str, tuple[str, str]]" = OrderedDict()  # key -> (kind, data)


class UserError(Exception):
    """An error whose message is safe to show to the user."""


# ---------------------------------------------------------------- whisper
_model = None
_model_lock = threading.Lock()


def get_model() -> WhisperModel:
    global _model
    with _model_lock:
        if _model is None:
            log.info("Loading Whisper model: %s", WHISPER_MODEL)
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        return _model


def transcribe(audio_path: str):
    segments, info = get_model().transcribe(
        audio_path,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    out = [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
    return out, info.language


# -------------------------------------------------------------- translate
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # optional: official Google Cloud Translation key
MAX_CHUNK = 450  # chars per request (MyMemory limit is ~500 bytes)
_cooldown: dict = {}  # provider name -> unix time until which it is skipped


def _tr_official(text: str, src: str) -> str:
    r = httpx.post(
        "https://translation.googleapis.com/language/translate/v2",
        params={"key": GOOGLE_API_KEY},
        data={"q": text, "target": "ar", "format": "text"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["data"]["translations"][0]["translatedText"]


def _tr_gtx(text: str, src: str) -> str:
    r = httpx.post(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": "auto", "tl": "ar", "dt": "t"},
        data={"q": text},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    r.raise_for_status()
    return "".join(part[0] for part in r.json()[0] if part and part[0])


def _tr_mymemory(text: str, src: str) -> str:
    r = httpx.get(
        "https://api.mymemory.translated.net/get",
        params={"q": text, "langpair": f"{src}|ar"},
        timeout=20,
    )
    r.raise_for_status()
    d = r.json()
    out = (d.get("responseData") or {}).get("translatedText") or ""
    if str(d.get("responseStatus")) != "200" or "MYMEMORY WARNING" in out:
        raise RuntimeError(str(d.get("responseDetails") or out)[:100])
    return out


def _translate_text(text: str, src: str) -> str:
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
                res = fn(text, src)
                if res and res.strip():
                    return res
            except Exception as e:
                log.warning("translate via %s failed (try %d): %s", name, attempt + 1, str(e)[:120])
                time.sleep(1)
        _cooldown[name] = time.time() + 600
    raise RuntimeError("all translation providers failed")


def translate_segments(segments, src: str = "auto"):
    """Translate to Arabic in a few small batches (one segment per line)."""
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
            out = _translate_text(joined, src).split("\n")
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
                result[i] = _translate_text(texts[i], src)
                ok += 1
            except Exception:
                pass
            time.sleep(0.4)

    if ok == 0:
        raise UserError("تعذرت الترجمة حالياً (خدمة الترجمة لا تستجيب). حاول بعد قليل.")
    return [(s, e, r) for (s, e, _), r in zip(segments, result)]


# -------------------------------------------------------------------- srt
def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments) -> str:
    blocks = []
    for i, (start, end, text) in enumerate(segments, start=1):
        blocks.append(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
    return "\n".join(blocks)


# ----------------------------------------------------------------- ffmpeg
async def run_cmd(args, cwd=None, timeout=900) -> bytes:
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
        raise UserError("استغرقت المعالجة وقتاً أطول من المسموح، جرّب مقطعاً أقصر.")
    if proc.returncode != 0:
        tail = err.decode(errors="ignore")[-1000:]
        raise RuntimeError(f"{args[0]} exited {proc.returncode}: {tail}")
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


async def burn_subtitles(video: Path, srt_name: str, workdir: Path, w: int, h: int) -> Path:
    scale = ""
    short_side = min(w, h)
    if short_side > 720:  # keep the file small and the encode fast
        f = 720 / short_side
        w, h = int(w * f) // 2 * 2, int(h * f) // 2 * 2
        scale = f"scale={w}:{h},"
    portrait = h > w
    font_size = 9 if portrait else 14   # relative to a 288px-high canvas
    margin_v = 35 if portrait else 15   # keep text above app UI / bottom edge
    style = (
        "FontName=Noto Sans Arabic,Bold=1,"
        f"FontSize={font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=2,Shadow=0,Alignment=2,"
        f"MarginV={margin_v},MarginL=20,MarginR=20"
    )
    out = workdir / "output.mp4"
    # srt is referenced by relative name and ffmpeg runs inside workdir -> no path escaping issues
    vf = f"{scale}subtitles={srt_name}:force_style='{style}'"
    await run_cmd(
        [
            "ffmpeg", "-y", "-i", str(video), "-vf", vf,
            "-c:v", "libx264", "-preset", "superfast", "-crf", "28", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out),
        ],
        cwd=str(workdir),
        timeout=1500,
    )
    return out


# --------------------------------------------------------------- download
def _ytdlp_download(url: str, workdir: Path, audio_only: bool = False) -> Path:
    opts = {
        "outtmpl": str(workdir / "input.%(ext)s"),
        # text-only mode needs just the audio -> much smaller/faster download
        "format": "ba/b" if audio_only else "bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
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
        raise UserError("تعذر تحميل الفيديو من هذا الرابط (قد يكون خاصاً أو محظوراً من السيرفر).")
    files = [p for p in workdir.glob("input.*") if p.suffix not in (".part", ".ytdl")]
    if not files:
        raise UserError("لم يتم العثور على فيديو في هذا الرابط.")
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
            raise UserError(
                "تيليجرام يمنع البوت من تنزيل ملفات أكبر من 20MB. "
                "ارفع الفيديو على أي موقع وأرسل لي الرابط بدلاً من الملف."
            )
        raise
    return dest


# --------------------------------------------------------------- pipeline
async def set_status(msg, text: str):
    try:
        await msg.edit_text(text)
    except TelegramError:
        pass


async def process(bot, chat_id: int, mode: str, kind: str, data: str, status):
    workdir = Path(tempfile.mkdtemp(prefix="subvid_"))
    try:
        async with JOB_SEM:
            await set_status(status, "⬇️ جاري تحميل الفيديو...")
            video = await fetch_input(bot, kind, data, workdir, audio_only=(mode == "txt"))

            w, h, duration = await probe(video)
            if duration and duration > MAX_MINUTES * 60:
                raise UserError(f"الحد الأقصى لمدة المقطع {MAX_MINUTES} دقائق.")

            await set_status(status, "🎧 جاري استخراج الصوت...")
            audio = workdir / "audio.wav"
            await run_cmd(
                ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(audio)]
            )

            await set_status(status, "📝 جاري تفريغ الكلام...")
            segments, lang = await asyncio.to_thread(transcribe, str(audio))
            if not segments:
                raise UserError("ما قدرت أرصد كلاماً واضحاً في المقطع.")

            if lang != "ar":
                await set_status(status, "🌐 جاري الترجمة إلى العربية...")
                segments = await asyncio.to_thread(translate_segments, segments, lang)

            srt_text = build_srt(segments)
            (workdir / "subs.srt").write_text(srt_text, encoding="utf-8")
            plain = "\n".join(t for _, _, t in segments)

            if mode == "txt":
                if len(plain) <= 3500:
                    await bot.send_message(chat_id, f"📄 الترجمة العربية:\n\n{plain}")
                else:
                    (workdir / "translation.txt").write_text(plain, encoding="utf-8")
                    with open(workdir / "translation.txt", "rb") as f:
                        await bot.send_document(chat_id, f, caption="📄 الترجمة العربية")
                with open(workdir / "subs.srt", "rb") as f:
                    await bot.send_document(chat_id, f, filename="subtitles_ar.srt",
                                            caption="ملف الترجمة المزمّنة (SRT)")
            else:
                await set_status(status, "🎬 جاري دمج الترجمة بالفيديو...")
                out = await burn_subtitles(video, "subs.srt", workdir, w, h)
                if out.stat().st_size > TG_UPLOAD_LIMIT:
                    with open(workdir / "subs.srt", "rb") as f:
                        await bot.send_document(
                            chat_id, f, filename="subtitles_ar.srt",
                            caption="⚠️ الفيديو الناتج أكبر من 50MB (حد تيليجرام للبوتات)، "
                                    "هذا ملف الترجمة بدلاً منه.",
                        )
                else:
                    await set_status(status, "📤 جاري رفع الفيديو...")
                    with open(out, "rb") as f:
                        await bot.send_video(chat_id, f, supports_streaming=True,
                                             caption="✅ تم دمج الترجمة العربية.")
        try:
            await status.delete()
        except TelegramError:
            pass
    except UserError as e:
        await set_status(status, f"⚠️ {e}")
    except Exception:
        log.exception("job failed")
        await set_status(status, "❌ حصل خطأ غير متوقع أثناء المعالجة. حاول مرة ثانية.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------- handlers
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "مرحباً بك في SubVid Ai 🎬\n\n"
        "أرسل لي رابط فيديو (يوتيوب، تيك توك، رابط مباشر) أو ملف فيديو، "
        "وأختار لك بين دمج الترجمة العربية بالفيديو أو استخراجها كنص."
    )


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
    if not kind:
        await msg.reply_text("أرسل رابط فيديو أو ملف فيديو.")
        return

    key = uuid.uuid4().hex[:10]
    JOBS[key] = (kind, data)
    while len(JOBS) > 300:
        JOBS.popitem(last=False)

    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🎬 دمج الترجمة العربية بالفيديو", callback_data=f"sub|{key}")],
            [InlineKeyboardButton("📄 استخراج الترجمة النصية", callback_data=f"txt|{key}")],
            [InlineKeyboardButton("إلغاء", callback_data=f"x|{key}")],
        ]
    )
    await msg.reply_text("وصلني ✅ اختر المطلوب:", reply_markup=keyboard)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except TelegramError:
        pass
    action, _, key = (q.data or "").partition("|")

    if action == "x":
        JOBS.pop(key, None)
        await q.edit_message_text("تم الإلغاء.")
        return

    job = JOBS.pop(key, None)  # pop => a double tap can't start the job twice
    if not job:
        await q.edit_message_text("انتهت صلاحية الطلب، أعد إرسال الفيديو.")
        return

    status = await q.edit_message_text("⏳ في قائمة الانتظار...")
    context.application.create_task(
        process(context.bot, q.message.chat_id, action, job[0], job[1], status)
    )


async def on_error(update, context: ContextTypes.DEFAULT_TYPE):
    log.error("handler error", exc_info=context.error)


async def post_init(app: Application):
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            log.critical("%s not found in PATH - deploy with the Dockerfile!", tool)
    await asyncio.to_thread(get_model)  # warm up so the first request is fast


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
    app.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE
            & ((filters.TEXT & ~filters.COMMAND) | filters.VIDEO | filters.Document.VIDEO),
            on_input,
        )
    )
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^(sub|txt|x)\|"))
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