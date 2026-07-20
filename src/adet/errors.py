class ExistingEquationError(Exception):
    """Raised when a the system is badly posed."""

    def __init__(self, message: str = 'The equation already exists in the system'):
        self.message = message
        super().__init__(message)
