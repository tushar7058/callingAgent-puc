from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

DateProvider = Callable[[], datetime]


class DailyLogFileHandler(logging.Handler):
    """Write log records to a file named for the current local date."""

    def __init__(
        self,
        log_dir: str | os.PathLike[str],
        *,
        date_provider: DateProvider | None = None,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        self.log_dir = Path(log_dir)
        self.date_provider = date_provider or datetime.now
        self.encoding = encoding
        self._current_date: str | None = None
        self._stream = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = self._stream_for_current_date()
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None
        super().close()

    def _stream_for_current_date(self):
        current_date = self.date_provider().strftime("%Y-%m-%d")
        if self._stream and self._current_date == current_date:
            return self._stream

        if self._stream:
            self._stream.close()

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date = current_date
        self._stream = (self.log_dir / f"{current_date}.log").open(
            "a", encoding=self.encoding
        )
        return self._stream


def setup_daily_file_logging(
    *,
    log_dir: str | os.PathLike[str] | None = None,
    level: str | int | None = None,
    date_provider: DateProvider | None = None,
) -> DailyLogFileHandler:
    """Attach a single daily file handler to the root logger."""

    resolved_log_dir = log_dir or os.getenv("AGENT_LOG_DIR") or "src/logs"
    resolved_level = _resolve_log_level(level or os.getenv("LOG_LEVEL") or "INFO")

    root_logger = logging.getLogger()
    root_logger.setLevel(min(root_logger.level or resolved_level, resolved_level))

    existing_handler = _existing_daily_handler(root_logger, resolved_log_dir)
    if existing_handler:
        existing_handler.setLevel(resolved_level)
        return existing_handler

    handler = DailyLogFileHandler(
        resolved_log_dir,
        date_provider=date_provider,
    )
    handler._aira_daily_file_handler = True
    handler.setLevel(resolved_level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
    return handler


def _existing_daily_handler(
    logger: logging.Logger, log_dir: str | os.PathLike[str]
) -> DailyLogFileHandler | None:
    resolved_log_dir = Path(log_dir).resolve()
    for handler in logger.handlers:
        if not getattr(handler, "_aira_daily_file_handler", False):
            continue
        if (
            isinstance(handler, DailyLogFileHandler)
            and handler.log_dir.resolve() == resolved_log_dir
        ):
            return handler
    return None


def _resolve_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    return getattr(logging, level.upper(), logging.INFO)
