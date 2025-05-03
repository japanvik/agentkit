# AgentKit Tests

This directory contains tests for the AgentKit framework. The tests are organized by component type:

- `agents/`: Tests for agent components
- `brains/`: Tests for brain components
- `memory/`: Tests for memory components

## Running Tests

To run all tests:

```bash
pytest
```

To run tests for a specific component:

```bash
pytest tests/agents/
pytest tests/brains/
pytest tests/memory/
```

To run a specific test file:

```bash
pytest tests/agents/test_base_agent.py
```

To run a specific test:

```bash
pytest tests/agents/test_base_agent.py::TestBaseAgent::test_initialization
```

## Test Coverage

To run tests with coverage:

```bash
pytest --cov=agentkit
```

To generate a coverage report:

```bash
pytest --cov=agentkit --cov-report=html
```

This will generate an HTML coverage report in the `htmlcov` directory.

## Writing Tests

When writing tests for AgentKit components:

1. Create test files in the appropriate directory (e.g., `tests/agents/` for agent tests)
2. Use pytest fixtures for test setup
3. Use descriptive test names that explain what is being tested
4. Use the `@pytest.mark.asyncio` decorator for tests that involve async functions
5. Mock external dependencies when appropriate
6. Test both success and failure cases
7. Test edge cases and boundary conditions
