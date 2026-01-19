#!/bin/bash

# Definition of ports
BACKEND_PORT=8000
FRONTEND_PORT=5174

echo "=================================================="
echo "      Model Lab: Dev Server Restarter"
echo "=================================================="

# Function to kill process on a port
kill_port() {
    PORT=$1
    PID=$(lsof -ti:$PORT)
    if [ -n "$PID" ]; then
        echo "Killing process $PID on port $PORT..."
        kill -9 $PID
    else
        echo "Port $PORT is free."
    fi
}

# 1. Cleanup
echo "[1/4] Cleaning up ports..."
kill_port $BACKEND_PORT
kill_port $FRONTEND_PORT

# 2. Activate Venv (Force existing venv)
echo "[2/4] Activating Python Environment..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated .venv"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated venv"
else
    echo "WARNING: No .venv or venv directory found!"
    echo "Python execution may fail or use system python."
fi

# 3. Start Backend
echo "[3/4] Starting Backend (Port $BACKEND_PORT)..."
export PYTHONPATH=$PYTHONPATH:.
# Running directly with python to ensure we use the activated venv
python server/main.py > server.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID (Logs: server.log)"

# 4. Start Frontend
echo "[4/4] Starting Frontend (Port $FRONTEND_PORT)..."
cd client
npm run dev -- --port $FRONTEND_PORT > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID (Logs: frontend.log)"
cd ..

# 5. Wait loop
echo "=================================================="
echo "Servers running!"
echo "Backend:  http://localhost:$BACKEND_PORT"
echo "Frontend: http://localhost:$FRONTEND_PORT"
echo "=================================================="
echo "Press Ctrl+C to stop both servers."

trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

echo "Streaming backend logs (server.log)..."
# Stream logs to console. Ctrl+C will kill tail, triggering the trap to kill servers.
tail -f server.log
