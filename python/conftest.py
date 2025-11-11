"""
pytest configuration and shared fixtures for TLA+ translation tests.

This file is automatically discovered by pytest and provides:
- Command-line options for model configuration
- Session-scoped fixtures for translator setup
- Test result reporting hooks
- Asyncio event loop configuration
"""

import asyncio
import logging
import os
import sys

# CRITICAL: Disable LiteLLM's background logging worker BEFORE importing translate.py
# This prevents "bound to a different event loop" errors in pytest-asyncio
# Must be configured before litellm is imported (which happens when translate.py is imported)

# Set environment variables that litellm checks during initialization
os.environ["LITELLM_LOG"] = "ERROR"  # Minimize logging
os.environ["LITELLM_DROP_DEBUG_LOGS"] = "True"

# Import litellm and configure it to disable async logging
import litellm

# Disable all async logging features
litellm.turn_off_message_logging = True
litellm.suppress_debug_info = True
litellm.set_verbose = False

# NUCLEAR FIX: Completely disable LoggingWorker by patching its methods
try:
    from litellm.litellm_core_utils.logging_worker import LoggingWorker
    
    # Patch _worker_loop to prevent it from accessing the queue
    async def no_op_worker_loop(self):
        """No-op replacement for LoggingWorker._worker_loop() to prevent event loop issues."""
        # Just return immediately, don't try to access the queue
        return
    
    # Patch start to do nothing
    def no_op_start(self):
        """No-op replacement for LoggingWorker.start()."""
        # Don't create any tasks or queues
        pass
    
    # Patch add_log to do nothing
    def no_op_add_log(self, *args, **kwargs):
        """No-op replacement for LoggingWorker.add_log()."""
        pass
    
    # Apply all patches
    LoggingWorker._worker_loop = no_op_worker_loop
    LoggingWorker.start = no_op_start
    LoggingWorker.add_log = no_op_add_log
    
    # Stop and destroy any existing worker instance
    if hasattr(LoggingWorker, '_instance') and LoggingWorker._instance is not None:
        worker = LoggingWorker._instance
        if hasattr(worker, '_task') and worker._task is not None:
            try:
                worker._task.cancel()
            except:
                pass
        LoggingWorker._instance = None
    
    print("✓ LoggingWorker successfully patched to prevent event loop issues")
        
except (ImportError, AttributeError, Exception) as e:
    # If we can't patch the worker, log it but continue
    print(f"Warning: Could not patch LoggingWorker: {e}")

import pytest

from translate import TLATranslator, detect_workspace_root, setup_environment

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--model",
        action="store",
        default="azure/gpt-4.1",
        help="LLM model to use (default: azure/gpt-4.1)"
    )
    parser.addoption(
        "--model-id",
        action="store",
        default=None,
        help="Model ID for AWS Bedrock"
    )
    parser.addoption(
        "--mcp-url",
        action="store",
        default="http://localhost:59071/mcp",
        help="MCP server URL (default: http://localhost:59071/mcp)"
    )
    parser.addoption(
        "--puzzle-file",
        action="store",
        default=None,
        help="Path to a specific puzzle file to test (e.g., puzzles/DieHard.md or DieHard.md)"
    )


@pytest.fixture(scope="session")
def workspace_root():
    """Get workspace root directory."""
    root = detect_workspace_root()
    logger.info(f"📂 Workspace root: {root}")
    return root


@pytest.fixture(scope="session")
def model(request):
    """Get model from command line option."""
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def model_id(request):
    """Get model ID from command line option."""
    return request.config.getoption("--model-id")


@pytest.fixture(scope="session")
def mcp_url(request):
    """Get MCP URL from command line option."""
    return request.config.getoption("--mcp-url")


@pytest.fixture(scope="session")
def puzzle_file_option(request):
    """Get puzzle file path from command line option."""
    return request.config.getoption("--puzzle-file")


# Note: event_loop fixture is automatically provided by pytest-asyncio
# with session scope via pytest.ini: asyncio_default_fixture_loop_scope = session

@pytest.fixture(scope="session", autouse=True)
async def configure_litellm_in_loop():
    """
    Configure litellm within the pytest event loop to prevent queue binding issues.
    
    This fixture runs automatically once per test session, inside the session-scoped event loop.
    It ensures that litellm's async resources are properly configured in the correct loop.
    """
    # Ensure litellm logging is disabled
    litellm.turn_off_message_logging = True
    litellm.suppress_debug_info = True
    
    # Aggressively stop any existing logging worker that might be bound to a different loop
    try:
        from litellm.litellm_core_utils.logging_worker import LoggingWorker
        
        # Stop and destroy any existing worker instance
        if hasattr(LoggingWorker, '_instance'):
            worker = LoggingWorker._instance
            if worker is not None:
                # Cancel the worker task
                if hasattr(worker, '_task') and worker._task is not None:
                    worker._task.cancel()
                    try:
                        await asyncio.wait_for(worker._task, timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError):
                        # RuntimeError might occur if task is in different loop
                        pass
                
                # Close the queue if it exists
                if hasattr(worker, '_queue') and worker._queue is not None:
                    try:
                        # Clear any pending items
                        while not worker._queue.empty():
                            try:
                                worker._queue.get_nowait()
                            except:
                                break
                    except:
                        pass
                
                # Destroy the instance
                LoggingWorker._instance = None
        
        # Ensure the start method is still patched to no-op
        LoggingWorker.start = lambda self: None
                
        logger.info("✓ LiteLLM logging worker cleaned up successfully")
    except Exception as e:
        logger.warning(f"Note: Could not fully clean up logging worker: {e}")
    
    yield
    
    # Cleanup after tests
    try:
        from litellm.litellm_core_utils.logging_worker import LoggingWorker
        if hasattr(LoggingWorker, '_instance') and LoggingWorker._instance is not None:
            worker = LoggingWorker._instance
            if hasattr(worker, '_task') and worker._task is not None:
                try:
                    worker._task.cancel()
                except:
                    pass
    except Exception:
        pass


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Verify litellm logging was disabled at module level
    # This prevents "bound to a different event loop" errors with pytest-asyncio
    logger.info(f"✓ LiteLLM logging configuration: turn_off_message_logging={litellm.turn_off_message_logging}")
    
    # Register custom markers
    config.addinivalue_line(
        "markers", "synthesis: mark test as a synthesis test"
    )
    config.addinivalue_line(
        "markers", "validation: mark test as a validation test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    
    # Setup environment for tests
    setup_environment()
    
    # Log test configuration
    logger.info("=" * 80)
    logger.info("PYTEST TEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model: {config.getoption('--model')}")
    if config.getoption('--model-id'):
        logger.info(f"Model ID: {config.getoption('--model-id')}")
    logger.info(f"MCP URL: {config.getoption('--mcp-url')}")
    logger.info("=" * 80)


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Add markers based on test names
    for item in items:
        # Mark synthesis tests
        if "synthesis" in item.name:
            item.add_marker(pytest.mark.synthesis)
        
        # Mark trace refinement tests  
        if "trace" in item.name:
            item.add_marker(pytest.mark.validation)
        
        # Mark full refinement tests
        if "full" in item.name:
            item.add_marker(pytest.mark.validation)


def pytest_runtest_setup(item):
    """Called before running each test."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"STARTING TEST: {item.name}")
    logger.info("=" * 80)


def pytest_runtest_teardown(item, nextitem):
    """Called after running each test."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"FINISHED TEST: {item.name}")
    logger.info("=" * 80)
    logger.info("")


def pytest_report_header(config):
    """Add custom header to pytest report."""
    return [
        "TLA+ Translation Test Suite",
        f"Model: {config.getoption('--model')}",
        f"MCP URL: {config.getoption('--mcp-url')}",
    ]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information to terminal output."""
    if exitstatus == 0:
        terminalreporter.write_sep("=", "🎉 All tests passed!", green=True, bold=True)
    else:
        terminalreporter.write_sep("=", "❌ Some tests failed", red=True, bold=True)

