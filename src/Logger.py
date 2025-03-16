"""

    Prompt into chatGPT: 'Create a Logger Python class'

"""


import logging


class Logger:
    def __init__(self, name: str, log_file: str = "app.log"):
        # Create a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set the default log level
        
        # Create a file handler to log messages to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # You can set file handler's level to DEBUG or INFO
        
        # Create a console handler to log messages to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # You can set console handler's level to INFO or WARNING
        
        # Create a formatter for the logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Set the formatter for both handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_debug(self, message: str):
        self.logger.debug(message)
    
    def log_info(self, message: str):
        self.logger.info(message)
    
    def log_warning(self, message: str):
        self.logger.warning(message)
    
    def log_error(self, message: str):
        self.logger.error(message)
    
    def log_critical(self, message: str):
        self.logger.critical(message)

