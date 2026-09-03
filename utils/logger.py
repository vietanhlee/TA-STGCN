import os
import sys


class TeeLogger:
    """
    Dual stream logger that redirects stdout to both console and logfile.
    """
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        log_dir = os.path.dirname(filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if hasattr(self, 'log') and not self.log.closed:
            self.log.close()
