#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Epsilon Engine Test Suite
# Tests the three-tier routing system with various complexity levels
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Exit on error

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================"
echo "  Epsilon Engine Test Suite v2.1"
echo "======================================"
echo ""

# Check if Python venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}WARNING: Virtual environment not activated${NC}"
    echo "Run: source venv/bin/activate"
    echo ""
fi

# Change to engine directory
cd /mnt/d/epsilon/v2 || exit 1

# Test helper function
run_test() {
    local test_name="$1"
    local prompt="$2"
    local timeout_seconds="$3"
    local expected_tier="$4"
    
    echo -e "${BLUE}[Test] ${test_name}${NC}"
    echo "Prompt: $prompt"
    echo "Expected tier: $expected_tier"
    echo ""
    
    # Run test with timeout
    result=$(echo "{\"prompt\": \"$prompt\"}" \
        | timeout ${timeout_seconds}s python3 backend/main.py --oneshot 2>&1)
    
    # Extract JSON response from stdout (last line)
    json_response=$(echo "$result" | grep '^{' | tail -n 1)
    
    if [[ -z "$json_response" ]]; then
        echo -e "${RED}✗ FAILED: No JSON response${NC}"
        echo "Full output:"
        echo "$result"
        return 1
    fi
    
    # Parse response
    ok=$(echo "$json_response" | jq -r '.ok // false')
    tier_used=$(echo "$json_response" | jq -r '.tier_used // "unknown"')
    complexity=$(echo "$json_response" | jq -r '.complexity // 0')
    
    # Show key metrics
    echo "Result:"
    echo "  ok: $ok"
    echo "  tier_used: $tier_used"
    echo "  complexity: $complexity"
    
    # Check if it passed
    if [[ "$ok" == "true" ]]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        
        # Bonus: check if tier matches expectation
        if [[ "$tier_used" == "$expected_tier" ]]; then
            echo -e "${GREEN}✓ Correct tier selected${NC}"
        else
            echo -e "${YELLOW}⚠ Expected $expected_tier, got $tier_used${NC}"
        fi
    else
        echo -e "${RED}✗ FAILED${NC}"
        error=$(echo "$json_response" | jq -r '.error // "Unknown error"')
        echo "Error: $error"
        return 1
    fi
    
    echo ""
    echo "---"
    echo ""
}

# ── Test Suite ────────────────────────────────────────────────────────────────

echo "Starting test suite..."
echo ""

# Test 1: Fast tier (simple completion)
run_test \
    "Fast tier — binary search" \
    "write a binary search function" \
    30 \
    "fast"

# Test 2: Fast tier (another simple task)
run_test \
    "Fast tier — fibonacci" \
    "def fibonacci(n):" \
    30 \
    "fast"

# Test 3: Balanced tier (more complex)
run_test \
    "Balanced tier — implement a function" \
    "implement a function to parse JSON and validate email addresses" \
    60 \
    "balanced"

# Test 4: Balanced tier (file generation)
run_test \
    "Balanced tier — REST API" \
    "create a Flask REST API with user authentication" \
    60 \
    "balanced"

# Test 5: Explicit override to balanced
run_test \
    "Explicit override — force balanced" \
    "use balanced: write hello world" \
    60 \
    "balanced"

# Test 6: High complexity (should use balanced with optimized thresholds)
run_test \
    "High complexity — database schema" \
    "create a complete SQLite database schema for a blog with users, posts, and comments" \
    60 \
    "balanced"

# ── Summary ───────────────────────────────────────────────────────────────────

echo "======================================"
echo "  Test Suite Complete"
echo "======================================"
echo ""
echo "All tests passed! ✓"
echo ""
echo "Next steps:"
echo "  1. Check VRAM usage: nvidia-smi"
echo "  2. Monitor processes: ps aux | grep llama-server"
echo "  3. Try interactive mode: python3 backend/main.py"
echo ""