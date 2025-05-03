#!/bin/bash

# Simple script to run tests for AgentKit

# Default options
COVERAGE=0
VERBOSE=0
SPECIFIC_TEST=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--coverage)
      COVERAGE=1
      shift
      ;;
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    -t|--test)
      SPECIFIC_TEST="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-c|--coverage] [-v|--verbose] [-t|--test TEST_PATH]"
      exit 1
      ;;
  esac
done

# Build the command
CMD="python -m pytest"

# Add options
if [ $VERBOSE -eq 1 ]; then
  CMD="$CMD -v"
fi

if [ $COVERAGE -eq 1 ]; then
  CMD="$CMD --cov=agentkit --cov-report=term --cov-report=html"
fi

# Add specific test if provided
if [ -n "$SPECIFIC_TEST" ]; then
  CMD="$CMD $SPECIFIC_TEST"
fi

# Run the command
echo "Running: $CMD"
eval $CMD

# Open coverage report if generated
if [ $COVERAGE -eq 1 ]; then
  echo "Coverage report generated in htmlcov/"
  if [[ "$OSTYPE" == "darwin"* ]]; then
    open htmlcov/index.html
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open htmlcov/index.html
  fi
fi
