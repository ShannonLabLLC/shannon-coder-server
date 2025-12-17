

# 🛡️ Shannon Coder Server

**The Open Source AI API Server for Red Teaming, Pentesting, and Security Research.**

Shannon Coder Server is a transparent, open-source API server designed to provide robust, confident, and unrestricted AI capabilities for security professionals. It serves as a drop-in replacement for OpenAI and Anthropic endpoints, modified to support advanced red teaming workflows, Not Safe For Work web projects, exploit analysis, and reliable function calling.

> **⚠️ Disclaimer:** This tool is designed for security research, red teaming, and educational purposes. Please use responsibly and ethically.



## ⚡ Key Capabilities

* **🔌 Dual SDK Compatibility:** Native support for both **OpenAI** and **Anthropic** SDKs. Just swap the `base_url`.
* **🔓 Unrestricted Research:** Optimized for security analysis, pentesting scenarios, and "confident" generation without standard refusals.
* **🛠️ Advanced Tool Use:** Full support for Function Calling (Tools), JSON Schema enforcement, and multi-turn agentic loops.
* **🔍 Built-in Web Search:** Automatic, real-time web search integration with source citations.
* **📋 Structured Outputs:** Guarantee valid JSON outputs for reliable data extraction and automation.

---

## 🚀 Quick Start

Get up and running in minutes. Shannon mirrors standard client libraries, so you can integrate it into your existing tools (like unmodified Claude Code or OpenAI-based pentesting scripts).

### 1. Endpoint Configuration

| Provider Compatibility | Base URL | Auth Header |
| --- | --- | --- |
| **OpenAI Compatible** | `https://shannon-ai.com/v1` | `Authorization: Bearer <KEY>` |
| **Anthropic Compatible** | `https://shannon-ai.com` | `x-api-key: <KEY>` |

### 2. Usage Examples

#### 🐍 Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_SHANNON_API_KEY",
    base_url="https://shannon-ai.com/v1"
)

response = client.chat.completions.create(
    model="shannon-balanced-grpo",
    messages=[
        {"role": "system", "content": "You are a senior security researcher."},
        {"role": "user", "content": "Analyze this code snippet for potential buffer overflow vulnerabilities."}
    ],
    max_tokens=1024
)

print(response.choices[0].message.content)

```

#### 🐜 Python (Anthropic SDK)

Shannon seamlessly accepts Anthropic-styled requests, making it compatible with tools like `claude-code`.

```python
import anthropic

client = anthropic.Anthropic(
    api_key="YOUR_SHANNON_API_KEY",
    base_url="https://shannon-ai.com"
)

response = client.messages.create(
    model="shannon-deep-dapo",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Generate a report on common XSS vectors."}
    ],
    # Shannon requires the standard version header
    extra_headers={"anthropic-version": "2023-06-01"} 
)

print(response.content[0].text)

```

---

## 🤖 Available Models

Shannon offers specialized models optimized for different security workflows.

| Model ID | Alias | Context | Best Use Case |
| --- | --- | --- | --- |
| `shannon-balanced-grpo` | **Balanced** | 128K | Everyday tasks, fast chat, Q&A, basic scripting. |
| `shannon-deep-dapo` | **Deep** | 128K | Complex reasoning, exploit analysis, deep research, code auditing. |

---

## 🛠️ Features Deep Dive

### Web Search Integration

Shannon includes a native `web_search` tool. You do not need to implement the search logic yourself; simply request current information, and the model will utilize its search capabilities.

```python
# The model will automatically trigger a search if asked about recent events
response = client.chat.completions.create(
    model="shannon-balanced-grpo",
    messages=[{"role": "user", "content": "What are the latest CVEs released for Nginx today?"}]
)

```

### Structured Outputs (JSON Schema)

Force the model to return data that adheres to a strict schema—essential for automated scanning pipelines.

```python
response = client.chat.completions.create(
    model="shannon-balanced-grpo",
    messages=[{"role": "user", "content": "Extract credentials found in this log dump."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "credential_extraction",
            "schema": {
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                    "password_hash": {"type": "string"},
                    "risk_level": {"type": "string", "enum": ["low", "critical"]}
                },
                "required": ["username", "risk_level"]
            }
        }
    }
)

```

---

## 🔐 Authentication & Errors

### Headers

* **OpenAI Style:** `Authorization: Bearer <your-key>`
* **Anthropic Style:** `x-api-key: <your-key>` + `anthropic-version: 2023-06-01`

### Common Error Codes

* `401 Unauthorized`: Invalid API Key.
* `402 Quota Exceeded`: Account balance or token limit reached.
* `429 Rate Limited`: Too many requests.
* `500 Internal Error`: Server-side issue (check status page).

---

## 🤝 Contributing

We believe in transparency for security tools.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/exploit-module`).
3. Commit your changes.
4. Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

### Would you like me to create a `CONTRIBUTING.md` file specifically focused on how developers can add new "tools" or "modules" to the server?
