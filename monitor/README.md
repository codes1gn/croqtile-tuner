# CroqTuner Monitor

A persistent monitoring service for GPU kernel tuning with an optional auto-wake feature to automatically dispatch opencode agents.

## Features

- **Real-time Dashboard**: Monitor tuning tasks, GPU status, and iteration progress
- **Auto-wake Toggle**: Enable/disable automatic opencode dispatching via web UI
  - **ON**: Scheduler auto-starts opencode for pending tasks (tuning mode)
  - **OFF**: Pure monitor mode - observe artifacts without starting new agents
- **SSE Events**: Real-time updates pushed to the UI
- **Results Compatibility**: Reads existing `results.tsv` and checkpoint files from croqtile-tuner

## Architecture

```
┌─────────────────┐      REST + SSE      ┌──────────────────────┐
│   React SPA     │◄────────────────────►│   FastAPI Backend     │
│   (Vite + TW)   │                      │                      │
└─────────────────┘                      │  ┌──────────────┐    │
                                         │  │  Scheduler    │    │
                                         │  │  (heartbeat)  │    │
                                         │  └──────┬───────┘    │
                                         │         │            │
                                         │  ┌──────▼───────┐    │
                                         │  │  Agent        │◄──┼── auto_wake_enabled
                                         │  │  (subprocess) │    │   (toggle)
                                         │  └──────┬───────┘    │
                                         └─────────┼────────────┘
                                                   │
                                         ┌─────────▼────────────┐
                                         │  opencode CLI        │
                                         │  (reads skills,      │
                                         │   writes artifacts)  │
                                         └──────────────────────┘
```

## Quick Start

```bash
cd /home/albert/workspace/croqtile-tuner/monitor

# Copy and edit environment config
cp .env.example .env

# Start both backend and frontend
./scripts/start.sh
```

Services:
- Backend API: http://localhost:8642
- Frontend UI: http://localhost:5173

## Run as systemd services

Service unit templates are provided in `monitor/deploy/systemd/`.

```bash
# Install units (root systemd)
sudo cp /home/albert/workspace/croqtile-tuner/monitor/deploy/systemd/croqtuner-backend.service /etc/systemd/system/
sudo cp /home/albert/workspace/croqtile-tuner/monitor/deploy/systemd/croqtuner-frontend.service /etc/systemd/system/

# Ensure backend venv/deps are ready
cd /home/albert/workspace/croqtile-tuner/monitor/backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Ensure frontend deps are ready
cd /home/albert/workspace/croqtile-tuner/monitor/frontend
npm install

# Enable + start
sudo systemctl daemon-reload
sudo systemctl enable --now croqtuner-backend.service croqtuner-frontend.service

# Check
systemctl status croqtuner-backend.service
systemctl status croqtuner-frontend.service
```

## Data Compatibility

The monitor reads data from the parent `croqtile-tuner` directory:

```
croqtile-tuner/
├── tuning/
│   └── <gpu>/
│       └── <dsl>/
│           ├── logs/<shape_key>/results.tsv
│           ├── checkpoints/<shape_key>.json
│           └── srcs/<shape_key>/iter<NNN>_<tag>.co
├── .claude/skills/
│   ├── croq-tune/
│   ├── base-tune/
│   └── ...
└── monitor/           ← This directory
    ├── backend/
    ├── frontend/
    └── data/monitor.db
```

## API Endpoints

### Tasks
- `GET /api/tasks` - List all tasks
- `POST /api/tasks` - Create a new task
- `GET /api/tasks/{id}` - Get task details
- `PATCH /api/tasks/{id}` - Update task status and/or per-task model assignment
- `DELETE /api/tasks/{id}` - Delete task
- `POST /api/tasks/{id}/retry` - Retry a failed task
- `POST /api/tasks/{id}/resume` - Resume from a specific iteration

### Settings
- `GET /api/settings/model` - Get model settings
- `GET /api/settings/auto-wake` - Get auto-wake toggle state
- `PATCH /api/settings/auto-wake` - Toggle auto-wake

### System
- `GET /api/health` - System health + GPU info
- `GET /api/events` - SSE event stream

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CROQTUNER_TUNING_DIR` | `../tuning` | Path to tuning artifacts |
| `CROQTUNER_SKILLS_DIR` | `../.claude/skills` | Path to skill definitions |
| `CROQTUNER_HOST` | `0.0.0.0` | Backend host |
| `CROQTUNER_PORT` | `8642` | Backend port |
| `CROQTUNER_HEARTBEAT_SEC` | `30` | Heartbeat interval |
| `CROQTUNER_CHOREO_HOME` | `/home/albert/workspace/croqtile` | Choreo compiler path |
| `CROQTUNER_MOCK_MODE` | `false` | Run without opencode |
