import os
import time
import subprocess
import nest_asyncio
import yt_dlp
import whisper
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from deep_translator import GoogleTranslator

nest_asyncio.apply()

# قراءة التوكن بأمان من إعدادات السيرفر
TOKEN = os.environ.get("TOKEN")

print("⏳ جاري تحميل نموذج الذكاء الاصطناعي...")
model = whisper.load_model("tiny")

def fmt_time(sec: float) -> str:
    ms = int((sec - int(sec)) * 1000)
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

async def translate_and_burn(in_vid: str, srt_file: str, out_vid: str, msg, update: Update):
    await msg.edit_text("🤖 جاري تفريغ الصوت وترجمته...")
    
    result = model.transcribe(in_vid, language="en", fp16=False)
    segments = [seg for seg in result.get("segments", []) if seg["text"].strip()]
    
    if not segments:
        await msg.edit_text("⚠️ لم يتم العثور على كلام واضح!")
        return

    translator = GoogleTranslator(source='en', target='ar')
    translated_texts = []
    
    for seg in segments:
        txt = seg["text"].strip().replace("\n", " ")
        try:
            tr = translator.translate(txt)
            translated_texts.append(tr if tr else txt)
        except Exception:
            translated_texts.append(txt)
        time.sleep(0.2)

    with open(srt_file, "w", encoding="utf-8") as f:
        for i, (seg, ar_text) in enumerate(zip(segments, translated_texts), 1):
            if ar_text:
                f.write(f"{i}\n{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}\n{ar_text}\n\n")

    await msg.edit_text("🎬 جاري دمج الترجمة...")
    cmd = [
        "ffmpeg", "-y", "-i", in_vid,
        "-vf", f"subtitles={srt_file}:force_style='Fontname=Amiri,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3'",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", out_vid
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    await msg.edit_text("📤 جاري إرسال الفيديو...")
    with open(out_vid, "rb") as v:
        await update.message.reply_video(video=v, caption="تمت الترجمة بواسطة @SubVidAiBot ✅")
    await msg.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("أهلاً بك في SubVid Ai 🎬✨\nأرسل رابط تيك توك أو يوتيوب وسأترجمه لك فوراً!")

async def handle_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text.strip()
    if not url.startswith(("http://", "https://")):
        return

    msg = await update.message.reply_text("🔗 جاري تحميل الفيديو...")
    in_vid = f"in_{update.message.id}.mp4"
    srt_file = f"sub_{update.message.id}.srt"
    out_vid = f"out_{update.message.id}.mp4"

    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': in_vid, 'quiet': True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        await translate_and_burn(in_vid, srt_file, out_vid, msg, update)
    except Exception as e:
        await msg.edit_text(f"⚠️ حدث خطأ: {e}")
    finally:
        for f in [in_vid, srt_file, out_vid]:
            if os.path.exists(f):
                os.remove(f)

app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_link))

print("🚀 البوت شغال الآن على السيرفر!")
app.run_polling(drop_pending_updates=False)
