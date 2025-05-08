import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

class ProcessingStatus:
    """Class to track and save processing status for recovery."""
    
    def __init__(self, status_file):
        """
        Initialize processing status tracker.
        
        Parameters:
        ----------
        status_file : str
            Path to the status file.
        """
        self.status_file = status_file
        self.processed_groups = set()
        self.failed_groups = set()
        self.load_status()
        
    def load_status(self):
        """Load processing status from file if it exists."""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    self.processed_groups = set(data.get('processed', []))
                    self.failed_groups = set(data.get('failed', []))
            except Exception as e:
                logging.warning(f"Failed to load status file: {e}")
                
    def save_status(self):
        """Save current processing status to file."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump({
                    'processed': list(self.processed_groups),
                    'failed': list(self.failed_groups),
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logging.error(f"Failed to save status file: {e}")
            
    def mark_processed(self, group_id):
        """Mark a group as processed."""
        self.processed_groups.add(group_id)
        self.save_status()
        
    def mark_failed(self, group_id):
        """Mark a group as failed."""
        self.failed_groups.add(group_id)
        self.save_status()
        
    def is_processed(self, group_id):
        """Check if a group has been processed."""
        return group_id in self.processed_groups
        
    def get_remaining(self, all_groups):
        """Get list of groups that still need processing."""
        all_group_ids = {self._get_group_id(g) for g in all_groups}
        return [g for g in all_groups if self._get_group_id(g) not in self.processed_groups]
        
    def _get_group_id(self, group):
        """Generate a consistent ID for a group."""
        if isinstance(group, list) and len(group) > 0:
            return sha256("".join(group).encode()).hexdigest()[:16]
        return None

