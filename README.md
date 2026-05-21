# Simulation Service

This project demonstrates how to interact with a voice agent served via a WebSocket connection. The agent accepts and returns base64-encoded audio, enabling real-time voice interactions.

## Project Structure

- `agent.py` — Starts a web server to interact with the voice agent.
- `sim_client.py` — Starts a WebSocket client to connect and interact with the voice agent.
- `.env.example` — Example environment variables required for the project.
- `pyproject.toml` — Project dependencies and configuration.

## Getting Started

### 1. Clone the Repository

```
git clone <your-repo-url>
cd simulation_service
```

### 2. Set Up the Environment Variables

Copy the example environment file and populate the required variables:

```
cp .env.example .env
```

Edit `.env` and fill in the values for:
- `DEEPGRAM_API_KEY`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`

### 3. Install Dependencies

It is recommended to use a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or use `pyproject.toml` with uv
```

### 4. Start the Voice Agent Server

```
python agent.py
```

### 5. Start the WebSocket Client

In a new terminal (with the virtual environment activated):

```
python sim_client.py
```

## How It Works

- The server (`agent.py`) hosts a voice agent accessible via WebSocket.
- The client (`sim_client.py`) connects to the server, sending and receiving base64-encoded audio data.
- This setup allows for real-time, bidirectional voice communication.

## Notes

- Ensure all required environment variables are set before running the server or client.
- The project is intended for demonstration and development purposes.

## License

[MIT](LICENSE)
