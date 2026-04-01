"""
backend/telegram/bot.py
========================
Telegram bot integration for Epsilon IDE Engine v2.

This lets you send coding requests from your phone via Telegram
and receive the generated code back — all processed on your local PC.

How it works:
  1. You send a message to your Telegram bot from your phone
  2. The bot receives it and passes it to the Epsilon orchestrator
  3. The orchestrator generates code using the local GPU
  4. The response is sent back to you on Telegram

Setup (one time):
  1. Open Telegram and search for @BotFather
  2. Send /newbot and follow the instructions
  3. Copy the token BotFather gives you
  4. Paste it into config.yaml under telegram_token

Security:
  - telegram_allowed_users in config.yaml restricts who can use the bot
  - Leave it empty [] to allow anyone who finds the bot
  - Add your Telegram user ID (a number) to restrict to only you
  - Find your user ID by messaging @userinfobot on Telegram

Commands the bot understands:
  /start   — welcome message
  /clear   — clear conversation memory
  /status  — show engine health (GPU temp, RAM, tokens/second)
  /help    — show available commands
  Any other message is treated as a coding request.
"""

import asyncio
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)


class EpsilonTelegramBot:
    """
    Telegram bot that forwards messages to the Epsilon orchestrator.

    Args:
        token:         Telegram bot token from @BotFather
        orchestrator:  Orchestrator instance
        memory:        ConversationMemory instance
        model:         ModelServer instance (for status checks)
        allowed_users: List of Telegram user IDs allowed to use the bot.
                       Empty list means everyone is allowed.
    """

    def __init__(self, token: str, orchestrator, memory, model,
                 allowed_users: list = None):
        self.token         = token
        self.orchestrator  = orchestrator
        self.memory        = memory
        self.model         = model
        self.allowed_users = allowed_users or []

        # Build the Telegram application
        self.app = Application.builder().token(token).build()
        self._register_handlers()
        print(f"[Telegram] Bot initialised")
        if self.allowed_users:
            print(f"[Telegram] Restricted to users: {self.allowed_users}")
        else:
            print("[Telegram] Open to all users (set telegram_allowed_users to restrict)")

    def _register_handlers(self) -> None:
        """Register all command and message handlers."""
        self.app.add_handler(CommandHandler("start",  self._cmd_start))
        self.app.add_handler(CommandHandler("clear",  self._cmd_clear))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("help",   self._cmd_help))
        # Any non-command text message → treat as coding request
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

    def _is_allowed(self, user_id: int) -> bool:
        """Check if this user is allowed to use the bot."""
        if not self.allowed_users:
            return True  # open to everyone
        return user_id in self.allowed_users

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Welcome message."""
        if not self._is_allowed(update.effective_user.id):
            await update.message.reply_text("Sorry, you are not authorised to use this bot.")
            return

        await update.message.reply_text(
            "Epsilon IDE Engine v2 is running on your PC.\n\n"
            "Send me any coding request and I will generate code for you.\n\n"
            "Examples:\n"
            "• write a quicksort function in Python\n"
            "• fix the bug: def add(a,b) return a+b\n"
            "• explain what a decorator does\n"
            "• create a FastAPI project with a users table\n\n"
            "Commands:\n"
            "/clear — clear conversation memory\n"
            "/status — show engine status\n"
            "/help — show this message"
        )

    async def _cmd_clear(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Clear conversation memory."""
        if not self._is_allowed(update.effective_user.id):
            return
        self.memory.clear()
        await update.message.reply_text("Conversation memory cleared.")

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Show engine health status."""
        if not self._is_allowed(update.effective_user.id):
            return

        await update.message.reply_text("Checking engine status...")

        try:
            import psutil
            ram_mb  = psutil.Process().memory_info().rss / (1024 ** 2)
            mem_stats = self.memory.stats()

            # Quick speed test
            speed = self.model.get_speed_stats()

            status = (
                f"Epsilon Engine v2 Status\n"
                f"{'─' * 30}\n"
                f"Model: running\n"
                f"Speed: {speed['tokens_per_second']} tok/s\n"
                f"Python RAM: {ram_mb:.0f} MB\n"
                f"Memory turns: {mem_stats['turns']}/{mem_stats['max_turns']}\n"
                f"Engine: ready"
            )
        except Exception as e:
            status = f"Status check failed: {e}"

        await update.message.reply_text(status)

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Show help message."""
        await self._cmd_start(update, ctx)

    async def _handle_message(self, update: Update,
                               ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle any text message as a coding request.

        Steps:
          1. Check authorisation
          2. Show "thinking" indicator
          3. Dispatch to the orchestrator
          4. Send the result back
        """
        if not self._is_allowed(update.effective_user.id):
            await update.message.reply_text("Not authorised.")
            return

        prompt = update.message.text.strip()
        if not prompt:
            return

        # Show typing indicator so the user knows it is working
        await ctx.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        print(f"[Telegram] Request from {update.effective_user.username}: {prompt[:50]}")

        try:
            # Run the orchestrator
            output = await self.orchestrator.dispatch(prompt)
            result = output["result"]
            task   = output["task_type"]
            files  = output.get("files_written", [])

            # Format the response for Telegram
            if files:
                # File writing task — show what was created
                response = f"Created {len(files)} files:\n"
                for f in files:
                    response += f"\n{f['file']} — {f['status']}"
                response += f"\n\n{result}"
            elif task in ("CODE_GEN", "DESCRIBE"):
                # Wrap code in a code block for nice formatting
                response = f"```python\n{result}\n```"
            else:
                response = result

            # Telegram has a 4096 character limit per message
            if len(response) > 4000:
                response = response[:4000] + "\n\n[truncated — response too long]"

            await update.message.reply_text(
                response,
                parse_mode="Markdown"
            )

        except Exception as e:
            print(f"[Telegram] Error handling message: {e}")
            await update.message.reply_text(f"Error: {e}")

    def run(self) -> None:
        """Start the bot in polling mode (blocking)."""
        print("[Telegram] Bot started — polling for messages")
        print("[Telegram] Open Telegram and message your bot")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)
