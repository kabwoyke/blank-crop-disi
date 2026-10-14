"""
oplog.py
--------
Operation log: every step (run, folder, backup, crop, blank removal, error) is
written to a CSV file (opens directly in Excel) and, optionally, to MySQL.

Logging never interrupts processing: if MySQL fails it is disabled for the rest
of the run and the problem is reported through `warn_cb`.
"""
import csv
import os
import sys
import threading
import uuid
from datetime import datetime

COLUMNS = ["timestamp", "run_id", "folder", "operation", "file", "status",
           "pages", "blank_pages", "details"]

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS operation_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ts DATETIME(3) NOT NULL,
    run_id VARCHAR(36) NOT NULL,
    folder VARCHAR(1024),
    operation VARCHAR(32) NOT NULL,
    file VARCHAR(1024),
    status VARCHAR(128),
    pages INT NULL,
    blank_pages INT NULL,
    details TEXT,
    INDEX idx_run (run_id),
    INDEX idx_ts (ts)
) CHARACTER SET utf8mb4
"""

INSERT_SQL = ("INSERT INTO operation_log "
              "(ts, run_id, folder, operation, file, status, pages, blank_pages, details) "
              "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)")


def default_log_dir():
    base = os.path.dirname(sys.executable if getattr(sys, "frozen", False)
                           else os.path.abspath(__file__))
    return os.path.join(base, "logs")


class OperationLogger:
    def __init__(self, log_dir=None, mysql=None, warn_cb=None):
        """mysql: dict(host, port, user, password, database) or None."""
        self.run_id = str(uuid.uuid4())
        self.warn_cb = warn_cb or (lambda m: None)
        self._lock = threading.Lock()

        log_dir = log_dir or default_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(
            log_dir, f"operations_{datetime.now():%Y%m%d_%H%M%S}.csv")
        # utf-8-sig so Excel reads non-ASCII file names correctly
        self._fh = open(self.csv_path, "w", newline="", encoding="utf-8-sig")
        self._csv = csv.writer(self._fh)
        self._csv.writerow(COLUMNS)
        self._fh.flush()

        self._db = None
        if mysql:
            self._connect(mysql)

    def _connect(self, cfg):
        try:
            import pymysql
            self._db = pymysql.connect(
                host=cfg["host"], port=int(cfg.get("port") or 3306),
                user=cfg["user"], password=cfg.get("password", ""),
                database=cfg["database"], charset="utf8mb4",
                autocommit=True, connect_timeout=10)
            with self._db.cursor() as cur:
                cur.execute(TABLE_SQL)
        except Exception as e:
            self._db = None
            self.warn_cb(f"MySQL logging disabled: {e}")

    @property
    def mysql_active(self):
        return self._db is not None

    def log(self, operation, folder="", file="", status="", pages=None,
            blank_pages=None, details=""):
        now = datetime.now()
        row = [now.isoformat(sep=" ", timespec="milliseconds"), self.run_id,
               str(folder), operation, str(file), status,
               "" if pages is None else pages,
               "" if blank_pages is None else blank_pages, details]
        with self._lock:
            try:
                self._csv.writerow(row)
                self._fh.flush()
            except Exception as e:
                self.warn_cb(f"CSV log write failed: {e}")
            if self._db is not None:
                try:
                    with self._db.cursor() as cur:
                        cur.execute(INSERT_SQL, (
                            now, self.run_id, str(folder)[:1024], operation,
                            str(file)[:1024], status[:128], pages, blank_pages, details))
                except Exception as e:
                    self.warn_cb(f"MySQL logging disabled: {e}")
                    try:
                        self._db.close()
                    except Exception:
                        pass
                    self._db = None

    def close(self):
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass
            if self._db is not None:
                try:
                    self._db.close()
                except Exception:
                    pass
                self._db = None
