# Auto-GPT Project

## Overview

Auto-GPT is an experimental project that leverages the capabilities of the Llama language model to create an autonomous AI agent. This project aims to explore the potential of Llama in performing tasks autonomously by chaining together its "thoughts" to achieve user-defined goals.

## Features

- **Internet Access**: The AI can perform searches and gather information from the web.
- **Memory Management**: Supports both long-term and short-term memory to store and recall information.
- **Text Generation**: Utilizes GPT-4 for generating human-like text responses.
- **File Storage**: Capable of storing and summarizing information in files.
- **API Integrations**: Supports integration with various APIs for extended functionalities.

## Installation

To install Auto-GPT, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/akemmanuel/Auto-GPT/Auto-GPT.git
    cd Auto-GPT
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Environment Variables**:
    Rename `.env.template` to `.env` and fill in the required API keys and settings.

## Usage

To run Auto-GPT, use the following command:

```bash
python -m autogpt
```