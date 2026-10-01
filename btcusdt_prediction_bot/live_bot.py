from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import websocket
import yaml

from features import make_features


class BybitBTCProbabilityBot:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.symbol = str(
            cfg.get("symbol", "BTCUSDT")
        ).upper()

        self.timezone = cfg.get(
            "timezone",
            "Asia/Kolkata"
        )

        self.threshold = float(
            cfg.get("min_confidence", 0.85)
        )

        # ==================================================
        # TELEGRAM
        # ==================================================

        telegram = cfg.get("telegram", {}) or {}

        self.telegram_enabled = bool(
            telegram.get("enabled", False)
        )

        self.telegram_token = str(
            telegram.get("bot_token", "")
        ).strip()

        self.telegram_chat_id = str(
            telegram.get("chat_id", "")
        ).strip()

        self.telegram_every_5m = bool(
            telegram.get(
                "send_every_5m",
                True
            )
        )

        # ==================================================
        # MODELS
        # ==================================================

        model_dir = Path(
            cfg.get("model_dir", "models")
        )

        model_5_path = (
            model_dir / "model_5m.joblib"
        )

        model_15_path = (
            model_dir / "model_15m.joblib"
        )

        if not model_5_path.exists():
            raise FileNotFoundError(
                f"Missing model: {model_5_path}\n"
                f"Run python train.py first."
            )

        if not model_15_path.exists():
            raise FileNotFoundError(
                f"Missing model: {model_15_path}\n"
                f"Run python train.py first."
            )

        self.models = {
            5: joblib.load(model_5_path),
            15: joblib.load(model_15_path),
        }

        # ==================================================
        # HISTORICAL INDEX DATA
        # ==================================================

        self.buffer = self._load_history()

        # ==================================================
        # LIVE INDEX PRICE STATE
        # ==================================================

        self.index_minutes: dict[pd.Timestamp, dict] = {}

        self.last_index_price: float | None = None

        self.last_index_ts: pd.Timestamp | None = None

        # Most recent market volume for each minute.
        self.market_volume: dict[pd.Timestamp, float] = {}

        # ==================================================
        # SIGNAL LOG
        # ==================================================

        self.signal_log = Path(
            cfg.get(
                "paper_log",
                "data/paper_signals.csv"
            )
        )

        self.signal_log.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.last_prediction_minute = None
        self.last_telegram_minute = None

        # ==================================================
        # CONNECTION STATUS
        # ==================================================

        self.received_ticker = False
        self.received_kline = False

    # ======================================================
    # LOAD HISTORICAL DATA
    # ======================================================

    def _load_history(self) -> pd.DataFrame:
        data_dir = Path(
            self.cfg.get(
                "data_dir",
                "data"
            )
        )

        cache_path = (
            data_dir
            / f"{self.symbol}_"
              f"{self.cfg.get('history_days', 90)}d_"
              f"index_1m.parquet"
        )

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Historical data file not found:\n"
                f"{cache_path}\n\n"
                f"Run python train.py first."
            )

        df = pd.read_parquet(
            cache_path
        )

        if not isinstance(
            df.index,
            pd.DatetimeIndex
        ):
            df.index = pd.to_datetime(
                df.index,
                utc=True
            )

        else:
            df.index = pd.to_datetime(
                df.index,
                utc=True
            )

        df = df.sort_index()

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"Historical data is missing columns: "
                f"{missing}"
            )

        df = df[required].copy()

        # Keep enough history for indicators.
        df = df.tail(3000)

        print(
            f"Loaded {len(df):,} historical "
            f"1m INDEX candles for live features."
        )

        print(
            f"History: "
            f"{df.index[0]} -> {df.index[-1]}"
        )

        return df

    # ======================================================
    # LIVE INDEX TICKER
    # ======================================================

    def on_ticker(self, message: dict):
        data = message.get(
            "data",
            {}
        )

        if isinstance(data, list):
            if not data:
                return

            data = data[0]

        if not isinstance(data, dict):
            return

        # --------------------------------------------------
        # IMPORTANT:
        # Bybit ticker can send snapshot + delta.
        # Delta may not contain indexPrice every time.
        # Therefore preserve the last known indexPrice.
        # --------------------------------------------------

        raw_index_price = data.get(
            "indexPrice"
        )

        if raw_index_price not in (
            None,
            "",
            "0"
        ):
            try:
                self.last_index_price = float(
                    raw_index_price
                )
            except (
                TypeError,
                ValueError
            ):
                pass

        if self.last_index_price is None:
            return

        # Outer message timestamp.
        raw_ts = message.get("ts")

        try:
            if raw_ts is None:
                timestamp = pd.Timestamp.now(
                    tz="UTC"
                )
            else:
                timestamp = pd.to_datetime(
                    int(raw_ts),
                    unit="ms",
                    utc=True
                )
        except (
            TypeError,
            ValueError
        ):
            timestamp = pd.Timestamp.now(
                tz="UTC"
            )

        minute = timestamp.floor("min")

        price = self.last_index_price

        state = self.index_minutes.get(
            minute
        )

        if state is None:
            self.index_minutes[minute] = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
        else:
            state["high"] = max(
                state["high"],
                price
            )

            state["low"] = min(
                state["low"],
                price
            )

            state["close"] = price

        self.last_index_ts = timestamp
        self.received_ticker = True

    # ======================================================
    # CLOSED MARKET KLINE
    # ======================================================

    def on_kline(self, message: dict):
        data = message.get(
            "data",
            []
        )

        if not isinstance(
            data,
            list
        ):
            return

        for candle in data:

            # We only process completed 1m candles.
            if candle.get(
                "confirm",
                False
            ) is not True:
                continue

            try:
                start = pd.to_datetime(
                    int(candle["start"]),
                    unit="ms",
                    utc=True
                )

                volume = float(
                    candle.get(
                        "volume",
                        0
                    )
                )

            except (
                KeyError,
                TypeError,
                ValueError
            ):
                continue

            self.market_volume[start] = volume

            self.received_kline = True

            self.finalize_minute(
                start
            )

    # ======================================================
    # BUILD FINAL 1M INDEX CANDLE
    # ======================================================

    def finalize_minute(
        self,
        start: pd.Timestamp
    ):
        state = self.index_minutes.pop(
            start,
            None
        )

        volume = self.market_volume.pop(
            start,
            None
        )

        if volume is None:
            volume = 0.0

        # --------------------------------------------------
        # If ticker data was slightly delayed and there is
        # no stored OHLC for this minute, use latest price
        # as fallback instead of crashing.
        # --------------------------------------------------

        if state is None:

            if self.last_index_price is None:
                print(
                    f"⚠️ {start.isoformat()} "
                    f"confirmed 1m candle but "
                    f"no INDEX price available."
                )
                return

            state = {
                "open": self.last_index_price,
                "high": self.last_index_price,
                "low": self.last_index_price,
                "close": self.last_index_price,
            }

            print(
                f"⚠️ Using latest INDEX price "
                f"fallback for "
                f"{start.strftime('%H:%M:%S UTC')}"
            )

        row = pd.DataFrame(
            [{
                "open": state["open"],
                "high": state["high"],
                "low": state["low"],
                "close": state["close"],
                "volume": volume,
            }],
            index=[start]
        )

        # Replace same minute if it already exists.
        self.buffer = (
            pd.concat([
                self.buffer.drop(
                    index=start,
                    errors="ignore"
                ),
                row,
            ])
            .sort_index()
            .tail(3000)
        )

        print(
            "\n"
            f"✅ CLOSED 1M INDEX CANDLE\n"
            f"Time : "
            f"{start.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Open : {state['open']:.2f}\n"
            f"High : {state['high']:.2f}\n"
            f"Low  : {state['low']:.2f}\n"
            f"Close: {state['close']:.2f}\n"
            f"Volume: {volume:.4f}"
        )

        self.make_prediction(
            start
        )

    # ======================================================
    # MODEL PREDICTION
    # ======================================================

    def predict(
        self,
        horizon: int
    ) -> float | None:

        frame = make_features(
            self.buffer
        )

        artifact = self.models[
            horizon
        ]

        feature_list = artifact.get(
            "features"
        )

        if not feature_list:
            raise RuntimeError(
                f"Model {horizon}m does not "
                f"contain feature list."
            )

        # Current final row only.
        x = frame[
            feature_list
        ].iloc[[-1]]

        # If indicators aren't ready.
        if x.isna().any(
            axis=None
        ):
            return None

        base_model = artifact[
            "base_model"
        ]

        calibrator = artifact[
            "calibrator"
        ]

        raw_probability = float(
            base_model.predict_proba(
                x
            )[0, 1]
        )

        # Logit transform.
        raw_probability = np.clip(
            raw_probability,
            1e-6,
            1 - 1e-6
        )

        z = np.log(
            raw_probability
            /
            (1.0 - raw_probability)
        ).reshape(
            -1,
            1
        )

        probability_up = float(
            calibrator.predict_proba(
                z
            )[0, 1]
        )

        return probability_up

    # ======================================================
    # TREND / VOLATILITY FILTER
    # ======================================================

    def check_filters(
        self,
        direction: str,
        frame: pd.DataFrame
    ):
        signal_cfg = (
            self.cfg.get(
                "signal",
                {}
            )
            or {}
        )

        reasons = []

        # ----------------------------------------------
        # Trend
        # ----------------------------------------------

        require_trend = bool(
            signal_cfg.get(
                "require_trend_filter",
                True
            )
        )

        if require_trend:

            if (
                "ema_21_dist" in frame.columns
                and
                "ema_50_dist" in frame.columns
            ):

                ema21 = float(
                    frame[
                        "ema_21_dist"
                    ].iloc[-1]
                )

                ema50 = float(
                    frame[
                        "ema_50_dist"
                    ].iloc[-1]
                )

                if direction == "UP":

                    if not (
                        ema21 > 0
                        and ema50 > 0
                    ):
                        reasons.append(
                            "trend not bullish"
                        )

                else:

                    if not (
                        ema21 < 0
                        and ema50 < 0
                    ):
                        reasons.append(
                            "trend not bearish"
                        )

        # ----------------------------------------------
        # ADX
        # ----------------------------------------------

        if "adx14" in frame.columns:

            try:
                adx = float(
                    frame[
                        "adx14"
                    ].iloc[-1]
                )
            except (
                TypeError,
                ValueError
            ):
                adx = np.nan

            min_adx = float(
                signal_cfg.get(
                    "min_adx",
                    16
                )
            )

            if np.isfinite(adx):

                if adx < min_adx:
                    reasons.append(
                        f"ADX {adx:.2f} "
                        f"< {min_adx:.2f}"
                    )

        # ----------------------------------------------
        # ATR
        # ----------------------------------------------

        if "atr14_pct" in frame.columns:

            try:
                atr_pct = float(
                    frame[
                        "atr14_pct"
                    ].iloc[-1]
                )
            except (
                TypeError,
                ValueError
            ):
                atr_pct = np.nan

            min_atr = float(
                signal_cfg.get(
                    "min_atr_pct",
                    0.00015
                )
            )

            max_atr = float(
                signal_cfg.get(
                    "max_atr_pct",
                    0.0045
                )
            )

            if np.isfinite(atr_pct):

                if atr_pct < min_atr:
                    reasons.append(
                        f"ATR too low "
                        f"{atr_pct:.6f}"
                    )

                elif atr_pct > max_atr:
                    reasons.append(
                        f"ATR too high "
                        f"{atr_pct:.6f}"
                    )

        return (
            len(reasons) == 0,
            reasons
        )

    # ======================================================
    # DECISION
    # ======================================================

    def get_decision(
        self,
        probability: float,
        other_probability: float | None,
        frame: pd.DataFrame
    ):

        direction = (
            "UP"
            if probability >= 0.5
            else "DOWN"
        )

        confidence = max(
            probability,
            1.0 - probability
        )

        # ----------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------

        if confidence < self.threshold:

            return (
                "NO TRADE",
                direction,
                confidence,
                [
                    f"confidence "
                    f"{confidence:.3f} "
                    f"< "
                    f"{self.threshold:.2f}"
                ]
            )

        # ----------------------------------------------
        # MODEL AGREEMENT
        # ----------------------------------------------

        signal_cfg = (
            self.cfg.get(
                "signal",
                {}
            )
            or {}
        )

        require_agreement = bool(
            signal_cfg.get(
                "require_model_agreement",
                True
            )
        )

        if (
            require_agreement
            and
            other_probability is not None
        ):

            this_side = (
                probability >= 0.5
            )

            other_side = (
                other_probability >= 0.5
            )

            if this_side != other_side:

                return (
                    "NO TRADE",
                    direction,
                    confidence,
                    [
                        "5m/15m model "
                        "disagreement"
                    ]
                )

        # ----------------------------------------------
        # MARKET FILTERS
        # ----------------------------------------------

        valid, filter_reasons = (
            self.check_filters(
                direction,
                frame
            )
        )

        if not valid:

            return (
                "NO TRADE",
                direction,
                confidence,
                filter_reasons
            )

        return (
            direction,
            direction,
            confidence,
            [
                "high-confidence setup"
            ]
        )

    # ======================================================
    # TELEGRAM
    # ======================================================

    def send_telegram(
        self,
        text: str
    ):

        if not self.telegram_enabled:
            return

        if (
            not self.telegram_token
            or
            not self.telegram_chat_id
        ):

            print(
                "⚠️ Telegram is enabled "
                "but bot_token/chat_id "
                "is missing."
            )

            return

        url = (
            "https://api.telegram.org/"
            f"bot{self.telegram_token}/"
            "sendMessage"
        )

        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:

            response = requests.post(
                url,
                json=payload,
                timeout=10
            )

            response.raise_for_status()

            result = response.json()

            if not result.get(
                "ok",
                False
            ):
                print(
                    "Telegram API error:",
                    result
                )
                return

            print(
                "📨 Telegram 5m update sent."
            )

        except requests.RequestException as exc:

            print(
                f"⚠️ Telegram error: "
                f"{exc}"
            )

    # ======================================================
    # BUILD TELEGRAM MESSAGE
    # ======================================================

    def build_telegram_message(
        self,
        timestamp: pd.Timestamp,
        p5: float,
        p15: float,
        action5: str,
        action15: str,
        reason5: list[str],
        reason15: list[str]
    ):

        local_time = (
            timestamp
            .tz_convert(
                self.timezone
            )
        )

        price = float(
            self.buffer[
                "close"
            ].iloc[-1]
        )

        direction5 = (
            "UP"
            if p5 >= 0.5
            else "DOWN"
        )

        direction15 = (
            "UP"
            if p15 >= 0.5
            else "DOWN"
        )

        confidence5 = max(
            p5,
            1.0 - p5
        )

        confidence15 = max(
            p15,
            1.0 - p15
        )

        agreement = (
            direction5 == direction15
        )

        emoji5 = (
            "🟢"
            if action5 == "UP"
            else
            "🔴"
            if action5 == "DOWN"
            else
            "⚪"
        )

        emoji15 = (
            "🟢"
            if action15 == "UP"
            else
            "🔴"
            if action15 == "DOWN"
            else
            "⚪"
        )

        return (
            "📊 <b>BTCUSDT 5-MIN UPDATE</b>\n\n"

            f"🕐 <b>Time:</b> "
            f"{local_time:%Y-%m-%d %H:%M:%S} IST\n"

            f"💰 <b>Bybit INDEX:</b> "
            f"${price:,.2f}\n\n"

            f"{emoji5} "
            f"<b>5M prediction: "
            f"{direction5}</b>\n"

            f"UP: "
            f"{p5 * 100:.2f}%\n"

            f"DOWN: "
            f"{(1.0 - p5) * 100:.2f}%\n"

            f"Confidence: "
            f"{confidence5 * 100:.2f}%\n"

            f"Decision: "
            f"<b>{action5}</b>\n\n"

            f"{emoji15} "
            f"<b>15M prediction: "
            f"{direction15}</b>\n"

            f"UP: "
            f"{p15 * 100:.2f}%\n"

            f"DOWN: "
            f"{(1.0 - p15) * 100:.2f}%\n"

            f"Confidence: "
            f"{confidence15 * 100:.2f}%\n"

            f"Decision: "
            f"<b>{action15}</b>\n\n"

            f"🤝 <b>5M / 15M agreement:</b> "
            f"{'YES' if agreement else 'NO'}\n"

            f"📡 <b>Data:</b> "
            f"LIVE 1M + Bybit INDEX\n"

            f"🧪 <b>Mode:</b> "
            f"PAPER / prediction only\n\n"

            f"5M filter: "
            f"{'; '.join(reason5[:2])}\n"

            f"15M filter: "
            f"{'; '.join(reason15[:2])}"
        )

    # ======================================================
    # RUN PREDICTION
    # ======================================================

    def make_prediction(
        self,
        timestamp: pd.Timestamp
    ):

        if (
            self.last_prediction_minute
            == timestamp
        ):
            return

        self.last_prediction_minute = (
            timestamp
        )

        try:
            frame = make_features(
                self.buffer
            )

            p5 = self.predict(5)
            p15 = self.predict(15)

        except Exception as exc:

            print(
                f"❌ Prediction error: "
                f"{exc}"
            )

            return

        if (
            p5 is None
            or
            p15 is None
        ):

            print(
                "⚠️ Indicators are not "
                "ready yet."
            )

            return

        action5, direction5, confidence5, reason5 = (
            self.get_decision(
                p5,
                p15,
                frame
            )
        )

        action15, direction15, confidence15, reason15 = (
            self.get_decision(
                p15,
                p5,
                frame
            )
        )

        # ==================================================
        # CONSOLE OUTPUT
        # ==================================================

        local_time = (
            timestamp
            .tz_convert(
                self.timezone
            )
        )

        price = float(
            self.buffer[
                "close"
            ].iloc[-1]
        )

        print(
            "\n"
            "==============================\n"
            "LIVE BTCUSDT PREDICTION\n"
            "==============================\n"
            f"Time: {local_time:%Y-%m-%d %H:%M:%S} IST\n"
            f"Index price: ${price:,.2f}\n\n"

            f"5M:\n"
            f"  UP   : {p5 * 100:.2f}%\n"
            f"  DOWN : {(1.0 - p5) * 100:.2f}%\n"
            f"  CONF : {confidence5 * 100:.2f}%\n"
            f"  DECISION: {action5}\n\n"

            f"15M:\n"
            f"  UP   : {p15 * 100:.2f}%\n"
            f"  DOWN : {(1.0 - p15) * 100:.2f}%\n"
            f"  CONF : {confidence15 * 100:.2f}%\n"
            f"  DECISION: {action15}\n\n"

            f"5M reason : "
            f"{'; '.join(reason5)}\n"

            f"15M reason: "
            f"{'; '.join(reason15)}\n"
            "=============================="
        )

        # ==================================================
        # PAPER SIGNAL LOG
        # ==================================================

        signal = {
            "signal_time_utc":
                timestamp.isoformat(),

            "signal_time_local":
                local_time.isoformat(),

            "symbol":
                self.symbol,

            "index_price":
                round(price, 4),

            "p5_up":
                round(p5, 6),

            "p5_down":
                round(1.0 - p5, 6),

            "p15_up":
                round(p15, 6),

            "p15_down":
                round(1.0 - p15, 6),

            "direction_5m":
                direction5,

            "direction_15m":
                direction15,

            "action_5m":
                action5,

            "action_15m":
                action15,

            "confidence_5m":
                round(confidence5, 6),

            "confidence_15m":
                round(confidence15, 6),

            "paper_mode":
                True,
        }

        file_exists = (
            self.signal_log.exists()
        )

        try:

            with self.signal_log.open(
                "a",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=signal.keys()
                )

                if not file_exists:
                    writer.writeheader()

                writer.writerow(signal)

        except Exception as exc:

            print(
                f"⚠️ Could not write "
                f"signal log: {exc}"
            )

        # ==================================================
        # TELEGRAM EVERY 5 MINUTES
        # ==================================================

        if (
            self.telegram_enabled
            and
            self.telegram_every_5m
            and
            timestamp.minute % 5 == 0
            and
            self.last_telegram_minute
            != timestamp
        ):

            self.last_telegram_minute = (
                timestamp
            )

            message = (
                self.build_telegram_message(
                    timestamp,
                    p5,
                    p15,
                    action5,
                    action15,
                    reason5,
                    reason15
                )
            )

            self.send_telegram(
                message
            )

    # ======================================================
    # CLEAN OLD LIVE STATE
    # ======================================================

    def cleanup_state(self):

        cutoff = (
            pd.Timestamp.now(
                tz="UTC"
            )
            - pd.Timedelta(
                minutes=10
            )
        )

        for minute in list(
            self.index_minutes.keys()
        ):

            if minute < cutoff:
                self.index_minutes.pop(
                    minute,
                    None
                )

        for minute in list(
            self.market_volume.keys()
        ):

            if minute < cutoff:
                self.market_volume.pop(
                    minute,
                    None
                )


# ==========================================================
# MAIN WEBSOCKET LOOP
# ==========================================================

def run():

    config_path = Path(
        "config.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "config.yaml not found."
        )

    cfg = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    bot = (
        BybitBTCProbabilityBot(
            cfg
        )
    )

    websocket_url = (
        "wss://stream.bybit.com/"
        "v5/public/linear"
    )

    reconnect_delay = 5

    while True:

        def on_open(ws):

            subscriptions = [
                f"kline.1.{bot.symbol}",
                f"tickers.{bot.symbol}",
            ]

            request = {
                "op": "subscribe",
                "args": subscriptions,
            }

            print(
                "\n"
                "======================================\n"
                "CONNECTING TO BYBIT\n"
                "======================================"
            )

            print(
                "URL:",
                websocket_url
            )

            print(
                "Subscribing:",
                subscriptions
            )

            ws.send(
                json.dumps(
                    request
                )
            )

            print(
                "✅ Connected to Bybit "
                "linear 1m stream + ticker indexPrice."
            )

            print(
                "✅ LIVE FEATURES use "
                "Bybit INDEX price."
            )

            print(
                "✅ No REST request "
                "required each minute."
            )

            print(
                "🧪 PAPER MODE: predictions only; "
                "no Bybit Odds order is submitted."
            )

        def on_message(
            ws,
            raw_message
        ):

            try:

                message = json.loads(
                    raw_message
                )

                # ------------------------------------------
                # SUBSCRIBE ACK
                # ------------------------------------------

                if message.get("op") == "subscribe":

                    print(
                        "📡 SUBSCRIBE RESPONSE:",
                        message
                    )

                    return

                # ------------------------------------------
                # PONG
                # ------------------------------------------

                if message.get("op") == "pong":
                    return

                # ------------------------------------------
                # TICKER
                # ------------------------------------------

                topic = message.get(
                    "topic",
                    ""
                )

                if topic == (
                    f"tickers.{bot.symbol}"
                ):

                    bot.on_ticker(
                        message
                    )

                    return

                # ------------------------------------------
                # KLINE
                # ------------------------------------------

                if topic == (
                    f"kline.1.{bot.symbol}"
                ):

                    bot.on_kline(
                        message
                    )

                    return

            except Exception as exc:

                print(
                    f"⚠️ Message processing error: "
                    f"{exc}"
                )

        def on_error(
            ws,
            error
        ):

            print(
                f"❌ WebSocket error: "
                f"{error}"
            )

        def on_close(
            ws,
            close_status_code,
            close_msg
        ):

            print(
                f"🔌 WebSocket closed: "
                f"{close_status_code} "
                f"{close_msg}"
            )

        def on_ping(
            ws,
            message
        ):

            # websocket-client automatically handles
            # standard ping/pong.
            pass

        try:

            ws = websocket.WebSocketApp(
                websocket_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_ping=on_ping,
            )

            ws.run_forever(
                ping_interval=20,
                ping_timeout=10,
                ping_payload="btc_odds_bot"
            )

        except KeyboardInterrupt:

            print(
                "\n🛑 Bot stopped by user."
            )

            break

        except Exception as exc:

            print(
                f"❌ WebSocket connection "
                f"exception: {exc}"
            )

        bot.cleanup_state()

        print(
            f"🔄 Reconnecting in "
            f"{reconnect_delay} seconds..."
        )

        try:
            time.sleep(
                reconnect_delay
            )
        except KeyboardInterrupt:
            print(
                "\n🛑 Bot stopped by user."
            )
            break


if __name__ == "__main__":
    run()