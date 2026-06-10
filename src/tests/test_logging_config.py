from __future__ import annotations

import logging
from datetime import datetime

from utils.logging_config import DailyLogFileHandler, setup_daily_file_logging


def test_daily_log_file_handler_writes_to_current_date_file(tmp_path):
    now = datetime(2026, 6, 10, 9, 30, 0)
    handler = DailyLogFileHandler(tmp_path, date_provider=lambda: now)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))

    logger = logging.getLogger("test.daily-log")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("hello logs")
    handler.close()

    assert (tmp_path / "2026-06-10.log").read_text(encoding="utf-8") == (
        "INFO:hello logs\n"
    )


def test_daily_log_file_handler_rolls_to_new_date(tmp_path):
    dates = [
        datetime(2026, 6, 10, 23, 59, 0),
        datetime(2026, 6, 11, 0, 1, 0),
    ]

    handler = DailyLogFileHandler(tmp_path, date_provider=lambda: dates.pop(0))
    handler.setFormatter(logging.Formatter("%(message)s"))

    record_a = logging.LogRecord("test", logging.INFO, "", 1, "first", None, None)
    record_b = logging.LogRecord("test", logging.INFO, "", 1, "second", None, None)
    handler.emit(record_a)
    handler.emit(record_b)
    handler.close()

    assert (tmp_path / "2026-06-10.log").read_text(encoding="utf-8") == "first\n"
    assert (tmp_path / "2026-06-11.log").read_text(encoding="utf-8") == "second\n"


def test_setup_daily_file_logging_is_idempotent(tmp_path):
    logger = logging.getLogger("test.setup-daily")
    logger.handlers = []

    handler_a = setup_daily_file_logging(
        log_dir=tmp_path,
        level="INFO",
        date_provider=lambda: datetime(2026, 6, 10, 12, 0, 0),
    )
    handler_b = setup_daily_file_logging(
        log_dir=tmp_path,
        level="INFO",
        date_provider=lambda: datetime(2026, 6, 10, 12, 0, 0),
    )

    try:
        assert handler_a is handler_b
    finally:
        root_logger = logging.getLogger()
        root_logger.removeHandler(handler_a)
        handler_a.close()


def test_setup_daily_file_logging_captures_agent_logger(tmp_path):
    handler = setup_daily_file_logging(
        log_dir=tmp_path,
        level="INFO",
        date_provider=lambda: datetime(2026, 6, 10, 12, 0, 0),
    )
    agent_logger = logging.getLogger("interview-agent")
    previous_level = agent_logger.level
    previous_propagate = agent_logger.propagate
    agent_logger.setLevel(logging.NOTSET)
    agent_logger.propagate = True

    try:
        agent_logger.info("agent lifecycle event")
        handler.close()
        log_text = (tmp_path / "2026-06-10.log").read_text(encoding="utf-8")
        assert "interview-agent" in log_text
        assert "agent lifecycle event" in log_text
    finally:
        agent_logger.setLevel(previous_level)
        agent_logger.propagate = previous_propagate
        root_logger = logging.getLogger()
        if handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        handler.close()
