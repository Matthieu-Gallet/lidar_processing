
@dataclass
class ProcessingOptions:
    """Configuration options for LiDAR processing."""
    keep_variables: Optional[List[str]] = None
    thin_radius: Optional[float] = None
    quality_levels: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default quality levels if none provided."""
        if self.quality_levels is None:
            self.quality_levels = [
                {"name": "high", "thin_radius": None},
                {"name": "medium", "thin_radius": 0.5},
                {"name": "low", "thin_radius": 1.0},
                {"name": "minimal", "thin_radius": 2.0}
            ]

