echo "Activating Virutal Environment...."
. /root/ey-chat-venv/bin/activate
sleep 1
export OPENAI_API_KEY=""
export GROQ_API_KEY=""
nohup  uvicorn app:app --log-level debug --reload --port 8002  > nohup_server.out 2>&1 &
# uvicorn app:app --log-level debug --reload --port 8002  