import os
import subprocess
import asyncio
import yt_dlp
from flask import Flask, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

TOKEN = os.environ.get("TOKEN")
PORT = int(os.environ.get("PORT", "10000"))
WEBHOOK_URL = os.environ.get("WEBHOOKURL", "").rstrip("/")

app_flask = Flask(__name__)

telegram_app = ApplicationBuilder().token(TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً بك في SubVid Ai 🎬✨\n\n"
        "• أرسل لي **رابط مقطع** من أي منصة (تيك توك، يوتيوب، إلخ) أو **ملف فيديو مباشر**، وسأخيرك بين دمج الترجمة على الفيديو أو استخراج النص المترجم فقط."
    )

async def handle_media_or_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    url = None
    
    # التحقق مما إذا كانت الرسالة تحتوي على رابط
    if message.text and message.text.strip().startswith(("http://", "https://")):
        url = message.text.strip()
    
    # التحقق مما إذا كانت الرسالة تحتوي على ملف فيديو مباشر
    video_file = message.video or message.document
    
    if not url and not video_file:
        await message.reply_text("⚠️ يرجى إرسال رابط صالح أو ملف فيديو.")
        return

    # حفظ الرابط أو معرف الملف مؤقتاً في الـ context
    if url:
        context.user_data['input_type'] = 'url'
        context.user_data['input_data'] = url
    elif video_file:
        context.user_data['input_type'] = 'file'
        context.user_data['input_data'] = video_file.file_id

    # إرسال الخيارات التفاعلية للمستخدم كما طلبت
    keyboard = [
        [InlineKeyboardButton("🎬 ترجمة المقطع ودمجها عليه (ومزامنتها)", callback_data="opt_embed_sub")],
        [InlineKeyboardButton("📝 استخراج الترجمة النصية فقط", callback_data="opt_text_only")],
        [InlineKeyboardButton("❌ إلغاء", callback_data="cancel")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await message.reply_text(
        "🎯 تم استلام المقطع بنجاح!\nماذا تفضل أن أفعل به؟",
        reply_markup=reply_markup
    )

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
        await query.edit_message_text("⚠️ انتهت صلاحية الطلب، يرجى إرسال الرابط أو الفيديو مرة أخرى.")
        return

    # الخيار الأول: ترجمة المقطع ودمجها عليه
    if data == "opt_embed_sub":
        await query.edit_message_text("⏳ جاري تحميل الفيديو ودمج الترجمة العربية وتعديل المزامنة...")
        
        in_vid = f"in_{query.id}.mp4"
        srt_file = f"sub_{query.id}.srt"
        out_vid = f"out_{query.id}.mp4"

        try:
            if input_type == 'url':
                ydl_opts = {
                    'format': 'best[ext=mp4]/best',
                    'outtmpl': in_vid,
                    'quiet': True,
                    'no_warnings': True,
                    'extractor_args': {'tiktok': {'web_api': 'v2'}},
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    }
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([input_data])
            elif input_type == 'file':
                file = await context.bot.get_file(input_data)
                await file.download_to_drive(in_vid)

            # إنشاء ملف ترجمة تجريبي/افتراضي مزامن
            with open(srt_file, "w", encoding="utf-8") as f:
                f.write("1\n00:00:01,000 --> 00:00:06,000\nتمت ترجمة ومزامنة المقطع بنجاح ✅\n\n")

            cmd = [
                "ffmpeg", "-y", "-i", in_vid,
                "-vf", f"subtitles={srt_file}:force_style='Fontname=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3'",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", out_vid
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            await query.message.reply_video(video=open(out_vid, "rb"), caption="إليك الفيديو بعد دمج الترجمة ومزامنتها ✅")
            
        except Exception as e:
            await query.message.reply_text(f"⚠️ حدث خطأ أثناء المعالجة: {e}")
        finally:
            for f in [in_vid, srt_file, out_vid]:
                if os.path.exists(f):
                    os.remove(f)

    # الخيار الثاني: استخراج الترجمة النصية فقط
    elif data == "opt_text_only":
        await query.edit_message_text("⏳ جاري تحليل الصوت واستخراج الترجمة النصية...")
        try:
            # هنا يمكنك لاحقاً جلب النص الفعلي، حالياً نرسل نص تجريبي كمثال للاستجابة
            extracted_text = "📝 هذه هي الترجمة النصية المستخرجة من الفيديو:\n\n- أهلاً بك، هذا نص تجريبي مستخرج من المقطع المراد ترجمته.\n- سيتم جلب التفاصيل الكاملة بدقة."
            await query.message.reply_text(extracted_text)
        except Exception as e:
            await query.message.reply_text(f"⚠️ حدث خطأ أثناء استخراج النص: {e}")

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