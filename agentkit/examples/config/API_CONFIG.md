# API Configuration Guide

This guide explains how to configure API settings for your agents. There are several ways to configure the API settings, with environment variables being the recommended approach for security.

## Configuration Options

### 1. Environment Variables (Recommended)

Create a `.env` file in your project root and set your API configuration:

```env
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
```

In your config file:
```json
{
    "api_config": {
        "use_env": true
    }
}
```

### 2. Direct Configuration

Not recommended for production use due to security concerns.

```json
{
    "api_config": {
        "api_base": "https://api.openai.com/v1",
        "api_key": "your-api-key-here"
    }
}
```

### 3. Local LLM Configuration (e.g., Ollama)

For local LLM deployments that don't require authentication:

```json
{
    "api_config": {
        "api_base": "http://localhost:11434"
    }
}
```

### 4. Azure OpenAI Configuration

For Azure OpenAI deployments:

```json
{
    "api_config": {
        "api_base": "https://your-resource-name.openai.azure.com",
        "api_key": "your-azure-api-key",
        "api_version": "2023-05-15"
    }
}
```

## Security Best Practices

1. Always use environment variables for sensitive information like API keys
2. Never commit your `.env` file to version control
3. Keep a `.env.example` file in version control to show required variables
4. Use different API keys for development and production environments

## Environment Variables

The following environment variables are supported:

- `OPENAI_API_BASE`: The base URL for your API endpoint
- `OPENAI_API_KEY`: Your API authentication key
- `AZURE_API_VERSION`: Required for Azure OpenAI deployments

You can set these variables in your `.env` file or your system's environment variables.

## Configuration Precedence

The system checks for API configuration in the following order:

1. Direct configuration in the agent's config file
2. Environment variables (if `use_env: true` is set)
3. Default values (e.g., `http://localhost:11434` for local deployments)

## Example Usage

See `example_api_config.json` for a complete example of an agent configuration with environment-based API settings.
