import threading
import csv
from pathlib import Path

class ThreadSafeCSVWriter:
    """Thread-safe CSV writer for concurrent result saving."""
    
    def __init__(self, csv_path: str, fieldnames: list):
        self.csv_path = Path(csv_path)
        self.fieldnames = fieldnames
        self.lock = threading.Lock()
        self._initialize_file()
    
    def _initialize_file(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def write_row(self, row_data: dict):
        """Write a single row to CSV in a thread-safe manner."""
        with self.lock:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row_data)


class ThreadSafeLogWriter:
    """Thread-safe log writer for concurrent logging."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.lock = threading.Lock()
    
    def write(self, message: str):
        """Write message to log file in a thread-safe manner."""
        with self.lock:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(message)
