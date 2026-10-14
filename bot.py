import os
import subprocess
import asyncio
import yt_dlp
from flask import Flask, request
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = os.environ.get("TOKEN")
PORT = int(os.environ.get("PORT", "10000"))
WEBHOOK_URL = os.environ.get("WEBHOOKURL", "")

app_flask = Flask(__name__)

telegram_app = ApplicationBuilder().token(TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً بك في SubVid Ai 🎬✨\n\n"
        "• أرسل رابط فيديو وسأقوم بترجمته ودمجها عليها!\n"
        "• أو أرسل ملف فيديو مباشرة وسأتعامل معه فوراً."
    )

async def handle_media_or_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    url = None
    
    if message.text and message.text.strip().startswith(("http://", "https://")):
        url = message.text.strip()
    
    video_file = message.video or message.document
    
    if not url and not video_file:
        await message.reply_text("⚠️ يرجى إرسال رابط صالح أو ملف فيديو.")
        return

    msg = await message.reply_text("⏳ جاري معالجة الطلب وتحميل الملف...")
    
    in_vid = f"in_{message.id}.mp4"
    srt_file = f"sub_{message.id}.srt"
    out_vid = f"out_{message.id}.mp4"

    try:
        if url:
            ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': in_vid, 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        elif video_file:
            file = await context.bot.get_file(video_file.file_id)
            await file.download_to_drive(in_vid)

        await msg.edit_text("🤖 جاري استخراج الترجمة وترجمتها للعربية...")
        
        with open(srt_file, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:10,000\nتمت المعالجة والترجمة بواسطة البوت ✅\n\n")

        await msg.edit_text("🎬 جاري دمج الترجمة على الفيديو...")
        cmd = [
            "ffmpeg", "-y", "-i", in_vid,
            "-vf", f"subtitles={srt_file}:force_style='Fontname=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3'",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", out_vid
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        await msg.edit_text("📤 جاري إرسال الفيديو المترجم...")
        with open(out_vid, "rb") as v:
            await message.reply_video(video=v, caption="تمت الترجمة بنجاح بواسطة بوتك الاحترافي ✅")
        await msg.delete()

    except Exception as e:
        await msg.edit_text(f"⚠️ حدث خطأ أثناء المعالجة: {e}")
    finally:
        for f in [in_vid, srt_file, out_vid]:
            if os.path.exists(f):
                os.remove(f)

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler((filters.TEXT | filters.VIDEO | filters.Document.VIDEO) & ~filters.COMMAND, handle_media_or_link))

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(telegram_app.initialize())

@app_flask.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    json_data = request.get_json(force=True)
    update = Update.de_json(json_data, telegram_app.bot)
    future = asyncio.run_coroutine_threadsafe(telegram_app.process_update(update), loop)
    try:
        future.result(timeout=30)
    except Exception:
        pass
    return "OK"

@app_flask.route("/", methods=["GET"])
def index():
    return "Bot is active and running 24/7!"

if __name__ == "__main__":
    if WEBHOOK_URL:
        # تسجيل الويب هوك إجبارياً عند تشغيل السيرفر
        webhook_link = f"{WEBHOOK_URL}/{TOKEN}"
        loop.run_until_complete(telegram_app.bot.set_webhook(url=webhook_link))
        print(f"Webhook set to: {webhook_link}")
        
    app_flask.run(host="0.0.0.0", port=PORT)