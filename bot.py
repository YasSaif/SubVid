import os
import subprocess
import asyncio
import yt_dlp
from flask import Flask, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

TOKEN = os.environ.get("TOKEN")
PORT = int(os.environ.get("PORT", "10000"))
WEBHOOK_URL = os.environ.get("WEBHOOKURL", "").rstrip("/")

app_flask = Flask(__name__)
telegram_app = ApplicationBuilder().token(TOKEN).build()

# تحميل نموذج Whisper المصغر (Tiny)
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً بك في SubVid Ai الاحترافي 🎬✨\n\n"
        "• أرسل لي **رابط مقطع** (تيك توك، يوتيوب، إلخ) أو **ملف فيديو مباشر**، وسأخيرك بين دمج الترجمة المزمنة أو استخراج النص!"
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

    if url:
        context.user_data['input_type'] = 'url'
        context.user_data['input_data'] = url
    elif video_file:
        context.user_data['input_type'] = 'file'
        context.user_data['input_data'] = video_file.file_id

    keyboard = [
        [InlineKeyboardButton("🎬 ترجمة المقطع ودمجها (ومزامنتها كالأفلام)", callback_data="opt_embed_sub")],
        [InlineKeyboardButton("📝 استخراج الترجمة النصية فقط", callback_data="opt_text_only")],
        [InlineKeyboardButton("❌ إلغاء", callback_data="cancel")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await message.reply_text(
        "🎯 تم استلام المقطع بنجاح!\nماذا تفضل أن أفعل به؟",
        reply_markup=reply_markup
    )

def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(int((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

async def process_audio_to_subtitles(audio_path):
    segments, info = whisper_model.transcribe(audio_path, beam_size=1, task="transcribe")
    
    srt_content = ""
    full_text_list = []
    
    for i, segment in enumerate(segments, start=1):
        start_str = seconds_to_srt_time(segment.start)
        end_str = seconds_to_srt_time(segment.end)
        english_text = segment.text.strip()
        
        if not english_text:
            continue
            
        try:
            arabic_text = GoogleTranslator(source='auto', target='ar').translate(english_text)
        except Exception:
            arabic_text = english_text
            
        full_text_list.append(arabic_text)
        srt_content += f"{i}\n{start_str} --> {end_str}\n{arabic_text}\n\n"
        
    return srt_content, "\n".join(full_text_list)

async def download_video_safely(input_type, input_data, out_filename):
    """تحميل الفيديو بطريقة آمنة ومستقرة تتخطى مشاكل المنصات"""
    if input_type == 'url':
        ydl_opts = {
            'format': 'best',
            'outtmpl': out_filename,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'extractor_args': {'tiktok': {'web_api': 'v2'}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([input_data])
    return os.path.exists(out_filename)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    input_type = context.user_data.get('input_type')
    input_data = context.user_data.get('input_data')

    if data == "cancel":
        await query.edit_message_text("❌ تم إلغاء العملية.")
        return

    if not input_data:
        await query.edit_message_text("⚠️ انتهت صلاحية الطلب، يرجى إرسال الرابط أو الفيديو من جديد.")
        return

    in_vid = f"in_{query.id}.mp4"
    audio_file = f"audio_{query.id}.mp3"
    srt_file = f"sub_{query.id}.srt"
    out_vid = f"out_{query.id}.mp4"

    try:
        if data == "opt_embed_sub":
            await query.edit_message_text("⏳ جاري تحميل الفيديو، تفريغ الصوت بالذكاء الاصطناعي، وترجمته ودمجه...")
            
            if input_type == 'url':
                success = await download_video_safely(input_type, input_data, in_vid)
                if not success:
                    await query.message.reply_text("⚠️ لم نتمكن من تحميل المقطع من الرابط، تأكد من صحته.")
                    return
            elif input_type == 'file':
                file = await context.bot.get_file(input_data)
                await file.download_to_drive(in_vid)

            subprocess.run(["ffmpeg", "-y", "-i", in_vid, "-vn", "-acodec", "libmp3lame", audio_file], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            srt_content, _ = await process_audio_to_subtitles(audio_file)
            
            if not srt_content.strip():
                await query.message.reply_text("⚠️ لم يتم رصد كلمات واضحة في المقطع لمعالجتها.")
                return

            with open(srt_file, "w", encoding="utf-8") as f:
                f.write(srt_content)

            cmd = [
                "ffmpeg", "-y", "-i", in_vid,
                "-vf", f"subtitles={srt_file}:force_style='Fontname=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3'",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", out_vid
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            await query.message.reply_video(video=open(out_vid, "rb"), caption="إليك الفيديو بعد دمج الترجمة ومزامنتها بنجاح ✅🎬")

        elif data == "opt_text_only":
            await query.edit_message_text("⏳ جاري تفريغ الصوت واستخراج الترجمة النصية وترجمتها...")
            
            if input_type == 'url':
                success = await download_video_safely(input_type, input_data, in_vid)
                if not success:
                    await query.message.reply_text("⚠️ لم نتمكن من تحميل المقطع من الرابط، تأكد من صحته.")
                    return
            elif input_type == 'file':
                file = await context.bot.get_file(input_data)
                await file.download_to_drive(in_vid)

            subprocess.run(["ffmpeg", "-y", "-i", in_vid, "-vn", "-acodec", "libmp3lame", audio_file], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            _, full_text = await process_audio_to_subtitles(audio_file)

            if full_text.strip():
                await query.message.reply_text(f"📝 **الترجمة النصية المستخرجة من المقطع:**\n\n{full_text[:4000]}")
            else:
                await query.message.reply_text("⚠️ لم يتم العثور على نصوص واضحة في الصوت.")

    except Exception as e:
        await query.message.reply_text(f"⚠️ حدث خطأ أثناء معالجة الطلب: {e}")
    finally:
        for f in [in_vid, audio_file, srt_file, out_vid]:
            if os.path.exists(f):
                os.remove(f)

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler((filters.TEXT | filters.VIDEO | filters.Document.VIDEO) & ~filters.COMMAND, handle_media_or_link))
telegram_app.add_handler(CallbackQueryHandler(button_callback))

@app_flask.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    try:
        json_data = request.get_json(force=True)
        update = Update.de_json(json_data, telegram_app.bot)
        
        async def process():
            async with telegram_app:
                await telegram_app.process_update(update)
                
        asyncio.run(process())
    except Exception as e:
        print(f"Error processing update: {e}")
    return "OK"

@app_flask.route("/", methods=["GET"])
def index():
    return "Bot is active and running 24/7!"

if __name__ == "__main__":
    if WEBHOOK_URL:
        webhook_link = f"{WEBHOOK_URL}/{TOKEN}"
        
        async def setup_webhook():
            async with telegram_app:
                await telegram_app.bot.delete_webhook(drop_pending_updates=True)
                await telegram_app.bot.set_webhook(url=webhook_link)
                print(f"Webhook successfully forced to: {webhook_link}")
                
        asyncio.run(setup_webhook())
        
    app_flask.run(host="0.0.0.0", port=PORT)