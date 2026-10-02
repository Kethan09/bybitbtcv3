# Telegram signal setup

The bot can send only high-confidence `UP` / `DOWN` signals to Telegram. `NO TRADE` is not sent by default.

## 1. Create your Telegram bot

1. Open Telegram and search for **@BotFather**.
2. Send `/newbot`.
3. Give the bot a name and username.
4. BotFather will give you a **bot token**. Keep it private.

## 2. Start your bot

Open the new bot in Telegram and send:

```text
/start
```

## 3. Get your chat ID

Open this URL in a browser after sending `/start`:

```text
https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
```

Replace `YOUR_BOT_TOKEN` with your real token.

In the JSON response, find:

```text
"chat":{"id":123456789
```

That number is your `chat_id`.

## 4. Configure the chat ID and token

```yaml
telegram:
  enabled: true
  bot_token: ""
  chat_id: "123456789"
  send_no_trade: false
```

Set the newly rotated token in PowerShell before starting the bot:

```powershell
$env:TELEGRAM_BOT_TOKEN = "YOUR_NEW_BOT_TOKEN"
python live_bot.py
```

The token must include the complete BotFather value, including the bot ID,
colon, and secret. The environment variable applies only to the current
PowerShell session.

For a private Telegram chat, the ID is normally a positive integer. For a group/channel it can have a different format.

## 5. Optional local AI commentary

Install Ollama for Windows, then download a small local model:

```powershell
ollama pull qwen2.5:3b
```

Enable the commentary in `config.yaml`:

```yaml
ollama:
  enabled: true
  base_url: http://localhost:11434
  model: qwen2.5:3b
  timeout_seconds: 10
```

Keep Ollama running while the bot is active. It adds a brief explanation
based only on the model probabilities and filters; it does not change the
prediction or make trading decisions. If Ollama is unavailable or times out,
the bot sends the regular Telegram update without AI commentary. No extra
Python package is required.

## 6. Run the bot

```powershell
python train.py
python live_bot.py
```

When a qualifying signal appears, Telegram will receive a message like:

```text
🟢 BTCUSDT UP SIGNAL

Expiry: 5 minutes
Index price: $108,250.00
UP: 91.20%
DOWN: 8.80%
Confidence: 91.20%
15M confirmation: UP 86.70% UP / 13.30% DOWN
Time: 2026-09-30T17:01:00+05:30
Mode: Prediction only
```

## Security

Do not post your Telegram bot token publicly or commit it to GitHub. For a production deployment, use environment variables or a secret manager instead of storing the token directly in `config.yaml`.
