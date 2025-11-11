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

# CRITICAL: Disable LiteLLM's background logging worker BEFORE importing translate.py
# This prevents "bound to a different event loop" errors in pytest
# Must be set before litellm is imported (which happens when translate.py is imported)
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "true"

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


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Note: LITELLM_DISABLE_LOGGING_WORKER is already set at module level (before imports)
    # but we verify it here for safety
    assert os.environ.get("LITELLM_DISABLE_LOGGING_WORKER") == "true", \
        "LiteLLM logging worker should be disabled"
    
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

