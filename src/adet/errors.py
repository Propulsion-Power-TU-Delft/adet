class ConstraintError(Exception):
    """Raised when a the system is badly posed."""

    def __init__(self, message: str = 'The object could not be solved'):
        self.message = message
        super().__init__(message)
