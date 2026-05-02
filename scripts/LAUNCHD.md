# macOS background runs (launchd)

Run the data pipeline on a schedule from your Mac so `data/historical_games.csv` stays fresh and the predictions log gets scored without opening Streamlit.

## 1) Paths

Replace `YOUR_USER` and, if needed, the project path and venv Python.

- Project: `/Users/YOUR_USER/Documents/nba-win-predictor`
- Interpreter: `/Users/YOUR_USER/Documents/nba-win-predictor/.venv/bin/python`
- Log directory: `logs/` under the project (created by `jobs/daily.py`)

## 2) Plist template

Save as `~/Library/LaunchAgents/com.nba-win-predictor.daily.plist` (label must be unique on your machine).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.nba-win-predictor.daily</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USER/Documents/nba-win-predictor/.venv/bin/python</string>
        <string>/Users/YOUR_USER/Documents/nba-win-predictor/jobs/daily.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USER/Documents/nba-win-predictor</string>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>15</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/YOUR_USER/Documents/nba-win-predictor/logs/launchd_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USER/Documents/nba-win-predictor/logs/launchd_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
```

This runs every day at **06:15** local time. Adjust `StartCalendarInterval` or switch to `StartInterval` (seconds) for a different cadence.

### Weekly full retrain (optional second job)

Duplicate the plist with a different `Label`, point `ProgramArguments` to the same `python` but add arguments:

```xml
<string>jobs/daily.py</string>
<string>--skip-fetch</string>
<string>--retrain</string>
```

Use a different clock (e.g. Monday 05:00) so you do not hammer `stats.nba.com` at the same minute as the fetch job.

## 3) Load and verify

```bash
launchctl unload ~/Library/LaunchAgents/com.nba-win-predictor.daily.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.nba-win-predictor.daily.plist
launchctl start com.nba-win-predictor.daily
```

Inspect:

- `logs/daily_pipeline.log` — step-by-step pipeline log from `jobs/daily.py`
- `logs/launchd_stdout.log` / `logs/launchd_stderr.log` — wrapper process I/O

## 4) Why not GitHub Actions for `fetch_data.py`?

The project README notes that `stats.nba.com` often blocks or throttles datacenter IPs. A scheduled job on **your home/office network** (this Mac) is the reliable default.

## 5) cron alternative

```cron
15 6 * * * cd /Users/YOUR_USER/Documents/nba-win-predictor && . .venv/bin/activate && python jobs/daily.py >> logs/cron.log 2>&1
```

Ensure `logs/` exists or mkdir in the crontab line.
