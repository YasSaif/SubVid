#!/bin/bash
# SubVid Ai - one-shot setup for an Ubuntu server (Oracle Cloud "Cloud-init script" box).
# Replace only the PUT_..._HERE texts between the single quotes, keep the quotes. Do NOT share this filled-in text with anyone.
TOKEN='PUT_BOT_TOKEN_HERE'
GROQ_API_KEY='PUT_GROQ_KEY_HERE'
ADMIN_ID='964933763'
DATABASE_URL='PUT_SUPABASE_SESSION_POOLER_URL_HERE'

exec > /var/log/subvid-setup.log 2>&1
set -x
say() { curl -s "https://api.telegram.org/bot${TOKEN}/sendMessage" -d chat_id="${ADMIN_ID}" --data-urlencode text="$1" >/dev/null; }
trap 'say "SubVid setup FAILED. Last lines:
$(tail -n 8 /var/log/subvid-setup.log | cut -c1-200)"' ERR
set -e

say "SubVid: Oracle server setup started (5-10 min)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git curl
curl -fsSL https://get.docker.com | sh

rm -rf /opt/subvid && git clone --depth 1 https://github.com/YasSaif/SubVid.git /opt/subvid
cd /opt/subvid
docker build -t subvid .

cat > /opt/subvid.env <<EOF
TOKEN=${TOKEN}
GROQ_API_KEY=${GROQ_API_KEY}
ADMIN_ID=${ADMIN_ID}
DATABASE_URL=${DATABASE_URL}
BURN_THREADS=4
EOF
chmod 600 /opt/subvid.env

# The container runs the code from /opt/subvid and pulls the newest version from GitHub every time it starts.
# So: /update in Telegram = restart = new code. Nightly the image is rebuilt too (in case libraries changed).
cat > /usr/local/bin/subvid-update <<'EOF'
#!/bin/bash
cd /opt/subvid && git pull -q
docker build -q -t subvid . >/dev/null
docker rm -f subvid >/dev/null 2>&1
docker run -d --name subvid --restart always --env-file /opt/subvid.env -v /opt/subvid:/app subvid \
  sh -c "git -C /app pull -q || true; exec python /app/bot.py"
EOF
chmod +x /usr/local/bin/subvid-update
/usr/local/bin/subvid-update
echo "30 4 * * * root /usr/local/bin/subvid-update >> /var/log/subvid-update.log 2>&1" > /etc/cron.d/subvid-update

say "SubVid: Oracle server is READY and the bot is running. Now suspend the Render service so they do not conflict."
