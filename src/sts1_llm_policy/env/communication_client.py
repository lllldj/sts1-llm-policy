import json
import sys
from collections.abc import Callable
from typing import TextIO


class CommunicationClient:
    """
    Thin wrapper around the stdin/stdout protocol used by
    CommunicationMod.

    Important:
    stdout is the command channel.
    Never write debug text to stdout outside send_command().
    """

    def __init__(
        self,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self._input = (
            input_stream
            if input_stream is not None
            else sys.stdin
        )

        self._output = (
            output_stream
            if output_stream is not None
            else sys.stdout
        )

        self._logger = logger

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger(message)

    def send_command(self, command: str) -> None:
        self._output.write(command + "\n")
        self._output.flush()

        self._log(
            f"sent command: {command}"
        )

    def receive_json(self) -> dict:
        for line in self._input:
            line = line.strip()

            if not line:
                continue

            self._log(
                f"received message: {len(line)} chars"
            )

            try:
                return json.loads(line)

            except json.JSONDecodeError as exc:
                self._log(
                    f"ignored non-JSON message: {exc}"
                )

        raise RuntimeError(
            "Communication channel closed before "
            "a JSON state was received"
        )

    def handshake(self) -> None:
        self.send_command("ready")