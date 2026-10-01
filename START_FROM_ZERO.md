# BTCUSDT Bybit Odds Prediction Bot — Clean Setup

This package is the clean starting point. It combines:
- 90-day Bybit BTCUSDT Index 1-minute historical training data
- 5-minute and 15-minute XGBoost probability models
- calibrated probabilities
- 85% high-confidence trade filter
- live Bybit WebSocket 1-minute timing + `indexPrice` ticker
- Telegram update every 5 minutes
- paper-only outcome tracking; no Bybit Odds order is submitted

## 1. Extract
Extract this ZIP to a NEW folder, for example:
`C:\BTCUSDT_ODDS_BOT\`

Do not mix files with your old bot folder.

## 2. PowerShell
Open PowerShell in:
`C:\BTCUSDT_ODDS_BOT\btcusdt_prediction_bot`

## 3. Create environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Telegram
Open `config.yaml` and set:
```yaml
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"
  send_no_trade: false
  send_every_5m: true
```
Never share the bot token.

## 5. Train
```powershell
python train.py
```
Successful output must include:
`Loaded 129,600 1m Bybit index candles`
and both `=== TRAIN 5 MINUTE ===` and `=== TRAIN 15 MINUTE ===`.

Models will be created in:
`models\model_5m.joblib`
`models\model_15m.joblib`

## 6. Start live paper mode
```powershell
python live_bot.py
```
The first lines should be:
`Connected to Bybit linear 1m stream + ticker indexPrice.`
`LIVE FEATURES use Bybit INDEX price; no REST request required each minute.`
`PAPER MODE: predictions only; no Bybit Odds order is submitted.`

Telegram will send one combined market update every 5 minutes. It will still display `UP`, `DOWN`, or `NO TRADE`; it does not require an 85% setup to send the 5-minute informational update.

## 7. What the live prediction means
At the close of each 1-minute candle:
- 5M model predicts the Index Price direction from the signal/entry reference to about 5 minutes later.
- 15M model predicts the Index Price direction from the signal/entry reference to about 15 minutes later.

## 8. Paper outcomes
Paper outcomes are written to:
`data\paper_outcomes.csv`

Do not use the historical 92–93% high-confidence test accuracy as a guaranteed live win rate. Evaluate live paper outcomes first.

## 9. Troubleshooting
If PowerShell does not show `(.venv)`, activate it:
```powershell
.\.venv\Scripts\activate
```
If `models\model_5m.joblib` is missing, run `python train.py` successfully first.
If you see repeated `getaddrinfo failed` or `ping/pong timed out`, it is a Bybit/network connectivity problem; do not trade during the outage.
