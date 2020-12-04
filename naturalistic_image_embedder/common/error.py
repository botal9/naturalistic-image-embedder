class ImageEmbedderError(Exception):
    """Base class for exceptions in this project."""
    def __init__(self, message):
        super().__init__(message)
