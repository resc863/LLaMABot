# LLaMABot

A Discord bot that interacts with users and uses the Ollama API for language model responses.

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Environment variable**

   Set a `discord_token` environment variable containing the Discord bot token.
   The bot reads this value when it starts:

   ```bash
   export discord_token="YOUR_DISCORD_BOT_TOKEN"
   ```

3. **Prepare the Ollama API**

   The `ChatOllama` cog expects an Ollama server running locally on
   `http://localhost:11434`. Install the Ollama CLI and start the server, for
   example:

   ```bash
   # Install ollama following the instructions from https://github.com/jmorganca/ollama
   ollama serve  # or `ollama run <model>`
   ```

   Ensure a supported model is available. The bot uses this server to send
   requests when generating responses.

## Running the bot

After setting the environment variable and starting the Ollama server, run:

```bash
python main.py
```

The bot will log in to Discord and start listening for messages.
