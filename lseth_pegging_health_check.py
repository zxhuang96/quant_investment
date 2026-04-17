import os
import json
import requests
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from zoneinfo import ZoneInfo
import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG
# =========================
LSETH_SYMBOL = "LSETH-USD"
ETH_SYMBOL = "ETH-USD"

COINBASE_URL = "https://api.coinbase.com/v2/prices/{}/spot"
EASTERN_TZ = ZoneInfo("America/New_York")

ROLLING_WINDOW = 24 * 7  # 7 days (hourly approx)
THRESHOLD_Z = 2.0

DAILY_RETURN_MEAN = 0.001
DAILY_RETURN_STD = 0.14

GSHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# =========================
# EMAIL CONFIG
# =========================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

EMAIL_SENDER = "zxhuang17@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = "zxhuang17@gmail.com"

# =========================
# GOOGLE SHEETS PERSISTENCE
# =========================
SHEET_HEADERS = ["timestamp", "lseth", "eth"]

def get_sheet():
    """Authenticate and return the first worksheet."""
    creds_json = os.getenv("GSHEET_CREDENTIALS")
    if not creds_json:
        raise ValueError("GSHEET_CREDENTIALS environment variable is not set")
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=GSHEET_SCOPES)
    gc = gspread.authorize(creds)
    sheet_id = os.getenv("GSHEET_ID")
    if not sheet_id:
        raise ValueError("GSHEET_ID environment variable is not set")
    return gc.open_by_key(sheet_id).sheet1


def load_history():
    """Load all rows from Google Sheet into a DataFrame."""
    ws = get_sheet()
    rows = ws.get_all_values()

    if len(rows) <= 1:
        # Sheet is empty or header only — write header if needed
        if len(rows) == 0:
            ws.append_row(SHEET_HEADERS)
        print("No existing data in sheet, starting fresh.")
        return pd.DataFrame(columns=SHEET_HEADERS)

    df = pd.DataFrame(rows[1:], columns=rows[0])  # skip header row
    df["lseth"] = df["lseth"].astype(float)
    df["eth"] = df["eth"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(EASTERN_TZ, ambiguous="infer")
    print(f"Loaded {len(df)} rows from Google Sheet.")
    return df


def save_latest_row(snap):
    """Append a single new row to Google Sheet."""
    ws = get_sheet()
    ws.append_row([
        snap["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        snap["lseth"],
        snap["eth"],
    ])


# =========================
# DATA FETCH
# =========================
def get_price(symbol):
    url = COINBASE_URL.format(symbol)
    r = requests.get(url)
    return float(r.json()["data"]["amount"])


def snapshot():
    return {
        "timestamp": datetime.now(EASTERN_TZ),
        "lseth": get_price(LSETH_SYMBOL),
        "eth": get_price(ETH_SYMBOL)
    }


def is_heartbeat_window(ts):
    return ts.hour in {7, 12, 18}


# =========================
# EMAIL FUNCTION
# =========================
def send_email(subject, content):
    msg = MIMEText(content)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    if not EMAIL_PASSWORD:
        raise ValueError("EMAIL_PASSWORD environment variable is not set")

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(EMAIL_SENDER, EMAIL_PASSWORD)
    server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
    server.quit()


# =========================
# FEATURES
# =========================
def compute_features(df):
    df = df.copy()

    df["ratio"] = df["lseth"] / df["eth"]

    df["mean"] = df["ratio"].rolling(ROLLING_WINDOW).mean()
    df["std"] = df["ratio"].rolling(ROLLING_WINDOW).std()
    df["zscore"] = (df["ratio"] - df["mean"]) / df["std"]

    # staking proxy (7d)
    df["ratio_lag"] = df["ratio"].shift(ROLLING_WINDOW)
    df["staking_7d"] = df["ratio"] / df["ratio_lag"] - 1
    df['staking_7d_zscore'] = (df['staking_7d'] - DAILY_RETURN_MEAN * ROLLING_WINDOW / 24) / (DAILY_RETURN_STD * np.sqrt(ROLLING_WINDOW / 24))

    return df


# =========================
# ALERT LOGIC
# =========================
def build_report(latest):
    timestamp_str = latest['timestamp'].strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"""
LSETH / ETH Monitor Report

Time (ET): {timestamp_str}

Price:
- LSETH: {latest['lseth']:.4f}
- ETH:   {latest['eth']:.4f}

Ratio:
- LSETH/ETH: {latest['ratio']:.6f}

Z-score:
- {latest['zscore']:.2f}

7D staking proxy:
- {latest.get('staking_7d', np.nan):.4%}
- Z-score: {latest.get('staking_7d_zscore', np.nan):.2f}

Alert:
- Hourly: {'YES' if abs(latest['zscore']) > THRESHOLD_Z else 'NO'}
- 7D staking proxy: {'YES' if abs(latest.get('staking_7d_zscore', np.nan)) > THRESHOLD_Z else 'NO'}

"""


def check_and_alert(df):
    latest = df.iloc[-1]

    if pd.isna(latest["zscore"]):
        return

    alert_flag = abs(latest["zscore"]) > THRESHOLD_Z or abs(latest.get("staking_7d_zscore", np.nan)) > THRESHOLD_Z
    report = build_report(latest)

    if alert_flag:
        send_email(
            subject="🚨 LSETH Depeg Alert",
            content=report
        )
    elif is_heartbeat_window(latest["timestamp"]):
        send_email(
            subject="LSETH Monitor (No Alert)",
            content=report
        )
    else:
        print(f"Skipping non-alert report outside ET heartbeat windows: {latest['timestamp']}")

    print(report)


# =========================
# MAIN (single run, scheduled by GitHub Actions)
# =========================
def run():
    snap = snapshot()
    save_latest_row(snap)        # append to Google Sheet immediately

    df = load_history()          # reload full history for feature computation
    df = compute_features(df)
    check_and_alert(df)


if __name__ == "__main__":
    run()