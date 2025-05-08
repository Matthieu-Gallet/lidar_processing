
import threading
import time
import logging
from typing import Optional


import psutil

class MemoryMonitor:
    """
    Class to monitor memory usage during processing.
    Provides real-time feedback and adaptive recommendations.
    """
    
    def __init__(self, threshold_mb=8000, critical_threshold_mb=9500, update_interval=1):
        """
        Initialize memory monitor.
        
        Parameters:
        ----------
        threshold_mb : int
            Warning memory threshold in MB.
        critical_threshold_mb : int
            Critical memory threshold in MB.
        update_interval : float
            Interval in seconds between memory checks.
        """
        self.threshold_mb = threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.peak_memory = 0
        self.current_memory = 0
        self.history = []
        self.warning_triggered = False
        self.critical_triggered = False
        self.logger = logging.getLogger("MemoryMonitor")
        
    def start(self):
        """Start monitoring memory usage."""
        self.running = True
        self.peak_memory = 0
        self.warning_triggered = False
        self.critical_triggered = False
        self.history = []
        self.thread = threading.Thread(target=self._monitor_memory)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop monitoring memory usage."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            
    def _monitor_memory(self):
        """Memory monitoring thread function."""
        while self.running:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.current_memory = memory_mb
            
            # Track memory history (last 10 data points)
            timestamp = time.time()
            self.history.append((timestamp, memory_mb))
            if len(self.history) > 10:
                self.history.pop(0)
                
            # Update peak memory
            if memory_mb > self.peak_memory:
                self.peak_memory = memory_mb
                
            # Check thresholds
            if memory_mb > self.critical_threshold_mb and not self.critical_triggered:
                self.logger.critical(
                    f"CRITICAL: Memory usage ({memory_mb:.2f} MB) exceeds critical threshold "
                    f"({self.critical_threshold_mb} MB)"
                )
                self.critical_triggered = True
                
            elif memory_mb > self.threshold_mb and not self.warning_triggered:
                self.logger.warning(
                    f"WARNING: Memory usage ({memory_mb:.2f} MB) exceeds threshold "
                    f"({self.threshold_mb} MB)"
                )
                self.warning_triggered = True
                
            # Reset triggers if memory goes back below thresholds
            if memory_mb < self.threshold_mb * 0.9:
                self.warning_triggered = False
                
            if memory_mb < self.critical_threshold_mb * 0.9:
                self.critical_triggered = False
                
            time.sleep(self.update_interval)
    
    def get_trend(self) -> float:
        """
        Calculate memory usage trend (MB/s).
        Positive value means memory is increasing.
        """
        if len(self.history) < 2:
            return 0.0
            
        # Calculate trend from first and last data points
        first_time, first_mem = self.history[0]
        last_time, last_mem = self.history[-1]
        time_diff = last_time - first_time
        
        if time_diff <= 0:
            return 0.0
            
        return (last_mem - first_mem) / time_diff
        
    def should_reduce_batch_size(self) -> bool:
        """Check if batch size should be reduced based on memory usage."""
        # Check if we're approaching the threshold with a rising trend
        trend = self.get_trend()
        approaching_threshold = self.current_memory > self.threshold_mb * 0.8
        
        return (self.warning_triggered or self.critical_triggered or 
                (approaching_threshold and trend > 10))  # Rising by 10MB/s
                
    def should_increase_batch_size(self) -> bool:
        """Check if batch size can be increased based on memory usage."""
        trend = self.get_trend()
        return (self.current_memory < self.threshold_mb * 0.5 and 
                trend < 5 and 
                not self.warning_triggered and 
                not self.critical_triggered)

