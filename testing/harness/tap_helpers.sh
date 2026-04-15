#!/usr/bin/env bash
# tap_helpers.sh — TAP-compatible test helpers for harness tests.
# Source this file in each test script.
#
# TAP = Test Anything Protocol (https://testanything.org/)
# Each test prints:   ok <N> <description>
#                 or: not ok <N> <description>

_TAP_COUNT=0
_TAP_FAIL=0
_PLAN=0

plan() {
  _PLAN="$1"
  echo "1..$1"
}

ok() {
  local desc="$1"
  local exitcode="${2:-0}"
  _TAP_COUNT=$((_TAP_COUNT + 1))
  if [ "$exitcode" -eq 0 ]; then
    echo "ok $_TAP_COUNT - $desc"
  else
    echo "not ok $_TAP_COUNT - $desc (exit $exitcode)"
    _TAP_FAIL=$((_TAP_FAIL + 1))
  fi
}

is() {
  local desc="$1"
  local got="$2"
  local expected="$3"
  _TAP_COUNT=$((_TAP_COUNT + 1))
  if [ "$got" = "$expected" ]; then
    echo "ok $_TAP_COUNT - $desc"
  else
    echo "not ok $_TAP_COUNT - $desc"
    echo "  # got:      $got"
    echo "  # expected: $expected"
    _TAP_FAIL=$((_TAP_FAIL + 1))
  fi
}

like() {
  local desc="$1"
  local string="$2"
  local pattern="$3"
  _TAP_COUNT=$((_TAP_COUNT + 1))
  if echo "$string" | grep -q "$pattern"; then
    echo "ok $_TAP_COUNT - $desc"
  else
    echo "not ok $_TAP_COUNT - $desc"
    echo "  # string:  $string"
    echo "  # pattern: $pattern"
    _TAP_FAIL=$((_TAP_FAIL + 1))
  fi
}

unlike() {
  local desc="$1"
  local string="$2"
  local pattern="$3"
  _TAP_COUNT=$((_TAP_COUNT + 1))
  if ! echo "$string" | grep -q "$pattern"; then
    echo "ok $_TAP_COUNT - $desc"
  else
    echo "not ok $_TAP_COUNT - $desc"
    echo "  # string should NOT contain: $pattern"
    echo "  # but got: $string"
    _TAP_FAIL=$((_TAP_FAIL + 1))
  fi
}

done_testing() {
  if [ "$_PLAN" -eq 0 ]; then
    echo "1..$_TAP_COUNT"
  fi
  if [ "$_TAP_FAIL" -gt 0 ]; then
    echo "# FAILED $_{TAP_FAIL} tests"
    exit 1
  fi
}
