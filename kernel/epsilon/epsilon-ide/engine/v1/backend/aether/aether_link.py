import asyncio
import json
import sys


class AetherLink:
    """
    Reads JSON requests from stdin line by line.
    Dispatches each through the orchestrator.
    Writes JSON responses to stdout.

    stdout is reserved ONLY for JSON responses.
    All logging uses stderr so it never pollutes the response stream.
    """

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    async def run(self) -> None:
        """Main loop — runs until stdin closes."""
        reader   = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        loop     = asyncio.get_event_loop()

        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        print("[AetherLink] Ready — listening on stdin", file=sys.stderr,
              flush=True)

        async for raw_bytes in reader:
            line = raw_bytes.decode('utf-8').strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                self._respond({'ok': False, 'error': f'Invalid JSON: {e}'})
                continue

            prompt = request.get('prompt', '').strip()
            if not prompt:
                self._respond({'ok': False, 'error': 'Empty prompt'})
                continue

            try:
                output = await self.orchestrator.dispatch(prompt)
                self._respond({
                    'ok':           True,
                    'result':       output['result'],
                    'task_type':    output['task_type'],
                    'valid_syntax': output['valid_syntax'],
                    'retried':      output['retried'],
                })
            except Exception as e:
                self._respond({'ok': False, 'error': str(e)})

    def _respond(self, data: dict) -> None:
        """
        Write JSON response directly to sys.stdout.

        We bypass print() entirely here because main.py overrides
        print() to redirect everything to stderr. This method must
        always write to stdout regardless of that override — it is
        the only thing in the entire engine that writes to stdout.
        """
        sys.stdout.write(json.dumps(data) + '\n')
        sys.stdout.flush()
