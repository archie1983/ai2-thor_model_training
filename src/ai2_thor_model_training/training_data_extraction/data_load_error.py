# Define Data Loading exception class
class DataLoadError(Exception):
    def __init__(self, message):
        super().__init__(message)
