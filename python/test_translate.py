#!/usr/bin/env python3
"""
pytest wrapper for TLA+ translation pipeline.

This test suite wraps the three main synthesis steps from translate.py:
1. Synthesize TLA+ specification from puzzle description
2. Synthesize trace refinement mapping to gold standard
3. Synthesize full refinement mapping to gold standard

Each synthesis step is a separate test with dependencies to ensure proper ordering.
Tests are parameterized by puzzle name for easy extensibility.

USAGE:
    # Run all tests
    pytest python/test_translate.py -v
    
    # Run a specific puzzle by file path
    pytest python/test_translate.py -v --puzzle-file puzzles/DieHard.md
    pytest python/test_translate.py -v --puzzle-file DieHard.md
    
    # Run a specific puzzle by name (using -k filter)
    pytest python/test_translate.py -v -k "DieHard"
    
    # Run a specific phase
    pytest python/test_translate.py -v -k "synthesis"
    
    # Run only synthesis phase for all puzzles
    pytest python/test_translate.py -v -k "synthesis"
    
    # Run with custom MCP URL
    pytest python/test_translate.py -v --mcp-url http://localhost:57812/mcp
    
    # Combine options: specific puzzle with custom MCP URL
    pytest python/test_translate.py -v --puzzle-file DieHard.md --mcp-url http://localhost:57812/mcp
    
    # Run with debug logging
    pytest python/test_translate.py -v --log-cli-level=DEBUG
    
    # Run with specific model
    pytest python/test_translate.py -v --model azure/gpt-4o
    
    # Run with AWS Bedrock
    pytest python/test_translate.py -v --model bedrock/anthropic.claude-sonnet-4-20250514-v1:0

REQUIREMENTS:
    Same as translate.py:
    - TLA+ MCP server running on http://localhost:59071/mcp
    - Java with tla2tools.jar available
    - API credentials (Azure OpenAI, AWS Bedrock, or GitHub Models)
    
ADDING NEW PUZZLES:
    Simply add the puzzle name to the PUZZLES list below. The puzzle file must exist
    in puzzles/{PuzzleName}.md and optionally gold/{PuzzleName}Gold.tla
"""

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import pytest

# Note: conftest.py configures litellm to avoid event loop issues before importing translate
# This prevents "bound to a different event loop" errors with pytest-asyncio's session-scoped loop
from translate import TLATranslator, detect_workspace_root, setup_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# PUZZLE CONFIGURATION
# ============================================================================
# Add new puzzles here - they will automatically get all three test phases
PUZZLES = [
    "DieHard",
    "CoffeeCan",
    "TowerOfHanoi",
    "CatInABox",
    "Prisoners",
    "RiverCrossingFlashlight",
]


def get_puzzle_name_from_file(puzzle_file_path: str) -> str:
    """Extract puzzle name from file path.
    
    Examples:
        "puzzles/DieHard.md" -> "DieHard"
        "DieHard.md" -> "DieHard"
        "/path/to/puzzles/DieHard.md" -> "DieHard"
    """
    from pathlib import Path
    return Path(puzzle_file_path).stem


def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters based on command-line options.
    
    If --puzzle-file is specified, only test that specific puzzle.
    Otherwise, test all puzzles in the PUZZLES list.
    """
    if "puzzle_files" in metafunc.fixturenames:
        puzzle_file_option = metafunc.config.getoption("--puzzle-file")
        
        if puzzle_file_option:
            # Extract puzzle name from file path
            puzzle_name = get_puzzle_name_from_file(puzzle_file_option)
            puzzles_to_test = [puzzle_name]
            logger.info(f"Testing specific puzzle from file: {puzzle_file_option} -> {puzzle_name}")
        else:
            # Use default list of all puzzles
            puzzles_to_test = PUZZLES
        
        metafunc.parametrize("puzzle_files", puzzles_to_test, indirect=True)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
async def translator(model, model_id, mcp_url, workspace_root):
    """Create and initialize TLATranslator instance."""
    setup_environment()
    
    translator = TLATranslator(model=model, model_id=model_id, mcp_url=mcp_url)
    
    # Verify TLC installation
    logger.info("🔧 Verifying TLC installation")
    tla2tools_path = workspace_root / 'tla2tools.jar'
    result = subprocess.run(
        ['java', '-XX:+UseParallelGC', '-jar', str(tla2tools_path)],
        capture_output=True, text=True, cwd=workspace_root
    )
    assert result.returncode == 1, "TLC installation missing or invalid"
    logger.info("✅ TLC installation verified")
    
    # Setup MCP connection
    await translator.setup_mcp_connection()
    
    return translator


@pytest.fixture(scope="function")
def puzzle_files(workspace_root, request):
    """Fixture to provide puzzle file paths based on test parameter."""
    # Get puzzle name from test parameter
    base_name = request.param
    
    puzzle_file = workspace_root / "puzzles" / f"{base_name}.md"
    assert puzzle_file.exists(), f"Puzzle file not found: {puzzle_file}"
    
    return {
        "puzzle_file": puzzle_file,
        "base_name": base_name,
        "tla_file": f"{base_name}.tla",
        "cfg_file": f"{base_name}.cfg",
        "gold_file": f"{base_name}Gold.tla",
        "trace_file": f"{base_name}Trace.tla",
        "trace_ref_file": f"{base_name}TraceRef.tla",
        "trace_cfg_file": f"{base_name}TraceRef.cfg",
        "ref_tla_file": f"{base_name}Ref.tla",
        "ref_cfg_file": f"{base_name}Ref.cfg",
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def assert_tlc_validation(
    translator: TLATranslator,
    tla_file: str,
    cfg_file: str,
    expected_exit_code: int,
    check_name: str,
    extra_args: Optional[list] = None
):
    """Assert that TLC validation succeeds with expected exit code.
    
    Args:
        translator: TLATranslator instance
        tla_file: TLA+ specification file
        cfg_file: TLC configuration file
        expected_exit_code: Expected TLC exit code (0 = success, 12 = counterexample)
        check_name: Description of the check for logging
        extra_args: Optional extra arguments for TLC
    """
    if extra_args is None:
        extra_args = []
    
    result = translator.run_tlc_and_validate(
        tla_file=tla_file,
        cfg_file=cfg_file,
        check_name=check_name,
        validate_exit_code=lambda code: code == expected_exit_code,
        extra_args=extra_args
    )
    
    assert result.returncode == expected_exit_code, (
        f"{check_name} expected exit code {expected_exit_code}, "
        f"got {result.returncode}\n"
        f"Output: {result.stdout}\n"
        f"Error: {result.stderr}"
    )


# ============================================================================
# PARAMETERIZED TESTS - PHASE 1: SYNTHESIS
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.dependency(name="synthesis[{puzzle_files}]")
async def test_synthesis(translator, puzzle_files, workspace_root, request):
    """Phase 1: Synthesize TLA+ specification from puzzle description.
    
    This test:
    - Reads the natural language puzzle description
    - Synthesizes a TLA+ specification
    - Generates a TLC configuration file
    - Validates that TLC finds a counterexample (the solution)
    
    Assertions:
    - 1.1: TLA+ specification file created
    - 1.2: TLC configuration file created
    - 1.3: TLC finds counterexample (exit code 12)
    """
    puzzle_name = puzzle_files["base_name"]
    logger.info("=" * 80)
    logger.info(f"{puzzle_name.upper()} - PHASE 1: SYNTHESIZE TLA+ SPECIFICATION")
    logger.info("=" * 80)
    
    synthesis_result = await translator.synthesize_tla_specification(
        puzzle_file=puzzle_files["puzzle_file"],
        tla_file=puzzle_files["tla_file"],
        cfg_file=puzzle_files["cfg_file"]
    )
    
    logger.info(f"✅ Synthesis completed for {puzzle_name}")
    
    # Assertion 1.1: TLA+ file was created
    tla_path = workspace_root / puzzle_files["tla_file"]
    assert tla_path.exists(), f"TLA+ file not created: {tla_path}"
    logger.info(f"✅ Assertion 1.1: TLA+ file exists")
    
    # Assertion 1.2: Config file was created
    cfg_path = workspace_root / puzzle_files["cfg_file"]
    assert cfg_path.exists(), f"Config file not created: {cfg_path}"
    logger.info(f"✅ Assertion 1.2: Config file exists")
    
    # Assertion 1.3: TLC finds counterexample (exit code 12)
    assert_tlc_validation(
        translator=translator,
        tla_file=puzzle_files["tla_file"],
        cfg_file=puzzle_files["cfg_file"],
        expected_exit_code=12,
        check_name="Synthesized specification validation"
    )
    logger.info(f"✅ Assertion 1.3: TLC found counterexample (exit code 12)")
    logger.info(f"✅ Phase 1 complete for {puzzle_name}")


# ============================================================================
# PARAMETERIZED TESTS - PHASE 2: TRACE REFINEMENT
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.dependency(name="trace[{puzzle_files}]", depends=["synthesis[{puzzle_files}]"])
async def test_trace_refinement(translator, puzzle_files, workspace_root, request):
    """Phase 2: Synthesize trace refinement mapping to gold standard.
    
    This test:
    - Copies the gold standard specification
    - Generates a trace from the counterexample
    - Creates a refinement mapping from trace to gold standard
    - Validates the refinement with TLC
    - Checks refinement postconditions
    
    Assertions:
    - 2.1: Trace file created
    - 2.2: Trace refinement file created
    - 2.3: Trace refinement config created
    - 2.4: Basic trace refinement validation passes
    - 2.5: Refinement postcondition passes
    - 2.6: Stats postcondition passes
    """
    puzzle_name = puzzle_files["base_name"]
    logger.info("=" * 80)
    logger.info(f"{puzzle_name.upper()} - PHASE 2: SYNTHESIZE TRACE REFINEMENT")
    logger.info("=" * 80)
    
    # Copy gold standard
    gold_source = workspace_root / "gold" / puzzle_files["gold_file"]
    if not gold_source.exists():
        pytest.skip(f"Gold standard file not found: {gold_source}")
    
    gold_dest = workspace_root / puzzle_files["gold_file"]
    shutil.copy2(gold_source, gold_dest)
    logger.info(f"📋 Copied gold standard: {puzzle_files['gold_file']}")
    
    # Synthesize trace refinement
    trace_result = await translator.synthesize_trace_refinement(
        tla_file=puzzle_files["tla_file"],
        gold_file=puzzle_files["gold_file"],
        trace_file=puzzle_files["trace_file"],
        trace_ref_file=puzzle_files["trace_ref_file"],
        trace_cfg_file=puzzle_files["trace_cfg_file"]
    )
    
    logger.info(f"✅ Trace refinement synthesis completed for {puzzle_name}")
    
    # Assertion 2.1: Trace file was created
    trace_path = workspace_root / puzzle_files["trace_file"]
    assert trace_path.exists(), f"Trace file not created: {trace_path}"
    logger.info(f"✅ Assertion 2.1: Trace file exists")
    
    # Assertion 2.2: Trace refinement file was created
    trace_ref_path = workspace_root / puzzle_files["trace_ref_file"]
    assert trace_ref_path.exists(), f"Trace refinement file not created: {trace_ref_path}"
    logger.info(f"✅ Assertion 2.2: Trace refinement file exists")
    
    # Assertion 2.3: Trace refinement config was created
    trace_cfg_path = workspace_root / puzzle_files["trace_cfg_file"]
    assert trace_cfg_path.exists(), f"Trace refinement config not created: {trace_cfg_path}"
    logger.info(f"✅ Assertion 2.3: Trace refinement config exists")
    
    # Assertion 2.4: Basic trace refinement validation (exit code 0)
    assert_tlc_validation(
        translator=translator,
        tla_file=puzzle_files["trace_ref_file"],
        cfg_file=puzzle_files["trace_cfg_file"],
        expected_exit_code=0,
        check_name="Trace refinement basic validation"
    )
    logger.info(f"✅ Assertion 2.4: Basic trace refinement validation")
    
    # Assertion 2.5: Trace refinement with Refinement postcondition
    gold_module = puzzle_files["base_name"] + "Gold"
    assert_tlc_validation(
        translator=translator,
        tla_file=puzzle_files["trace_ref_file"],
        cfg_file=puzzle_files["trace_cfg_file"],
        expected_exit_code=0,
        check_name="Trace refinement with Refinement postcondition",
        extra_args=['-postcondition', f'{gold_module}!Refinement']
    )
    logger.info(f"✅ Assertion 2.5: Trace refinement with Refinement postcondition")
    
    # Assertion 2.6: Trace refinement with Stats postcondition
    assert_tlc_validation(
        translator=translator,
        tla_file=puzzle_files["trace_ref_file"],
        cfg_file=puzzle_files["trace_cfg_file"],
        expected_exit_code=0,
        check_name="Trace refinement with Stats postcondition",
        extra_args=['-postcondition', f'{gold_module}!Stats']
    )
    logger.info(f"✅ Assertion 2.6: Trace refinement with Stats postcondition")
    logger.info(f"✅ Phase 2 complete for {puzzle_name}")


# ============================================================================
# PARAMETERIZED TESTS - PHASE 3: FULL REFINEMENT
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.dependency(name="full[{puzzle_files}]", depends=["trace[{puzzle_files}]"])
async def test_full_refinement(translator, puzzle_files, workspace_root, request):
    """Phase 3: Synthesize full refinement mapping to gold standard.
    
    This test:
    - Creates a full refinement mapping from synthesized spec to gold standard
    - Generates TLC configuration for the refinement
    - Validates the refinement with TLC
    - Checks refinement postconditions
    
    Assertions:
    - 3.1: Full refinement file created
    - 3.2: Full refinement config created
    - 3.3: Basic full refinement validation passes
    - 3.4: Refinement postcondition passes
    - 3.5: Stats postcondition passes
    """
    puzzle_name = puzzle_files["base_name"]
    logger.info("=" * 80)
    logger.info(f"{puzzle_name.upper()} - PHASE 3: SYNTHESIZE FULL REFINEMENT")
    logger.info("=" * 80)
    
    full_result = await translator.synthesize_full_refinement(
        tla_file=puzzle_files["tla_file"],
        gold_file=puzzle_files["gold_file"],
        ref_tla_file=puzzle_files["ref_tla_file"],
        ref_cfg_file=puzzle_files["ref_cfg_file"]
    )
    
    logger.info(f"✅ Full refinement synthesis completed for {puzzle_name}")
    
    # Assertion 3.1: Full refinement file was created
    ref_tla_path = workspace_root / puzzle_files["ref_tla_file"]
    assert ref_tla_path.exists(), f"Full refinement file not created: {ref_tla_path}"
    logger.info(f"✅ Assertion 3.1: Full refinement file exists")
    
    # Assertion 3.2: Full refinement config was created
    ref_cfg_path = workspace_root / puzzle_files["ref_cfg_file"]
    assert ref_cfg_path.exists(), f"Full refinement config not created: {ref_cfg_path}"
    logger.info(f"✅ Assertion 3.2: Full refinement config exists")
    
    # Assertion 3.3: Basic full refinement validation (exit code 0)
    assert_tlc_validation(
        translator=translator,
        tla_file=puzzle_files["ref_tla_file"],
        cfg_file=puzzle_files["ref_cfg_file"],
        expected_exit_code=0,
        check_name="Full refinement basic validation"
    )
    logger.info(f"✅ Assertion 3.3: Basic full refinement validation")
    
    # Assertion 3.4: Full refinement with Refinement postcondition
    gold_module = puzzle_files["base_name"] + "Gold"
    assert_tlc_validation(
        translator=translator,
        tla_file=puzzle_files["ref_tla_file"],
        cfg_file=puzzle_files["ref_cfg_file"],
        expected_exit_code=0,
        check_name="Full refinement with Refinement postcondition",
        extra_args=['-postcondition', f'{gold_module}!Refinement']
    )
    logger.info(f"✅ Assertion 3.4: Full refinement with Refinement postcondition")
    
    # Assertion 3.5: Full refinement with Stats postcondition
    assert_tlc_validation(
        translator=translator,
        tla_file=puzzle_files["ref_tla_file"],
        cfg_file=puzzle_files["ref_cfg_file"],
        expected_exit_code=0,
        check_name="Full refinement with Stats postcondition",
        extra_args=['-postcondition', f'{gold_module}!Stats']
    )
    logger.info(f"✅ Assertion 3.5: Full refinement with Stats postcondition")
    
    logger.info("=" * 80)
    logger.info(f"🎉 ALL TESTS PASSED FOR {puzzle_name}")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Allow running directly with python
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])
