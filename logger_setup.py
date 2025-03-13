import os
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional
from functools import wraps
import time
from datetime import datetime

class LoggerSetup:
    """
    Handles the setup and configuration of logging for the GovChat API system.
    Implements a singleton pattern to ensure only one logger instance exists.
    """
    
    _instance: Optional[logging.Logger] = None
    
    @classmethod
    def get_logger(cls, logger_name: str = 'govchat_api') -> logging.Logger:
        """
        Get or create a logger instance using singleton pattern.
        
        Args:
            logger_name (str): Name of the logger (default: 'govchat_api')
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if cls._instance is None:
            cls._instance = cls._setup_logger(logger_name)
        return cls._instance
    
    @staticmethod
    def _setup_logger(logger_name: str) -> logging.Logger:
        """
        Setup the logger with file rotation and formatting.
        
        Args:
            logger_name (str): Name of the logger
            
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logs directory outside repository
        log_dir = Path.home() / 'spesland_logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Create log file path with timestamp
        timestamp = datetime.now().strftime('%Y%m')
        log_file = log_dir / f'{logger_name}_{timestamp}.log'
        
        # Create rotating file handler
        file_handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=30,  # Rotate every 30 days
            backupCount=1,  # Keep 1 backup file (1 month retention)
            encoding='utf-8'
        )
        
        # Set log format
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            
            # Add console handler for development environment
            if os.getenv('ENVIRONMENT', '').lower() == 'development':
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            logger.info(f"Logger initialized. Writing logs to: {log_file}")
        
        return logger

def log_function_call(func):
    """
    Decorator to log function entry, exit, and execution time.
    Also logs any exceptions that occur during function execution.
    
    Usage:
        @log_function_call
        def your_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = LoggerSetup.get_logger()
        function_name = func.__name__
        logger.info(f"Entering function: {function_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Exiting function: {function_name} - Execution time: {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Error in {function_name} after {execution_time:.2f}s: {str(e)}", 
                exc_info=True
            )
            raise
    return wrapper

def log_request(func):
    """
    Decorator to log API request details.
    
    Usage:
        @log_request
        def your_api_endpoint():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = LoggerSetup.get_logger()
        request_time = datetime.now()
        
        try:
            # Log request details
            logger.info(f"API Request at {request_time.isoformat()}")
            if hasattr(func, '__name__'):
                logger.info(f"Endpoint: {func.__name__}")
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log response
            execution_time = (datetime.now() - request_time).total_seconds()
            logger.info(f"Request completed in {execution_time:.2f}s")
            
            return result
        except Exception as e:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            raise
    return wrapper

# Create default logger instance
logger = LoggerSetup.get_logger()

# Example usage:
if __name__ == '__main__':
    # Test the logger
    logger.info("Test log message")
    
    # Test the function decorator
    @log_function_call
    def test_function():
        logger.info("Inside test function")
        time.sleep(1)  # Simulate some work
        
    # Test the request decorator
    @log_request
    def test_api_endpoint():
        logger.info("Processing API request")
        time.sleep(0.5)  # Simulate request processing
        return {"status": "success"}
    
    test_function()
    test_api_endpoint()