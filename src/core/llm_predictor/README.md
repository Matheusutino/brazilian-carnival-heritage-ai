# 🤖 LLM Predictor Module

This module provides a unified interface to interact with different Large Language Model (LLM) providers. It uses the Factory design pattern to make it easy to create and use provider-specific predictors.

## 📁 Module Structure

```
llm_predictor/
├── __init__.py              # Public interface and exports
├── llm_predictor.py         # Abstract base class (ABC)
├── llm_factory.py           # Factory pattern for instantiation
└── nvidia_predictor.py      # NVIDIA API implementation
```

## 🏗️ Architecture

### Design Pattern: Abstract Factory + Strategy

The module uses a combination of patterns:
- **Abstract Base Class (ABC)**: `LLMPredictor` defines the contract
- **Factory Pattern**: `LLMFactory` creates appropriate instances
- **Strategy Pattern**: Each provider implements the same interface

```
┌─────────────────┐
│  LLMPredictor   │ (Abstract Base Class)
│   <<abstract>>  │
└────────┬────────┘
         │
┌────────▼────────┐
│ NVIDIAPredictor │ (Implemented)
└─────────────────┘
```

---

## 📄 Detailed Modules

### 1. `llm_predictor.py` - Abstract Base Class

Description: Defines the interface that all LLM implementations must follow.

Main Class:
```python
class LLMPredictor(ABC):
    @abstractmethod
    def predict(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int
    ) -> str:
        """Generate a response from the language model."""
        pass
```

predict() parameters:
- `system_prompt` (`str`): System instructions that set model behavior
- `user_prompt` (`str`): User input or query
- `temperature` (`float`): Controls randomness (0.0 = deterministic, 1.0 = creative)
- `max_output_tokens` (`int`): Maximum token limit for the response

Return: `str` — Model-generated response

---

### 2. `llm_factory.py` - Factory Pattern

Description: Centralizes predictor creation logic and simplifies switching between providers.

Main Class:
```python
class LLMFactory:
    @staticmethod
    def get_predictor(llm_provider: str, model_name: str) -> LLMPredictor:
        """Create and return an LLM predictor for the specified provider."""
```

Supported Providers:
- NVIDIA — `"nvidia"` — ✅ Implemented

Usage Example:
```python
from src.core.llm_predictor import LLMFactory

# Create NVIDIA predictor
predictor = LLMFactory.get_predictor(
    llm_provider="nvidia",
    model_name="meta/llama-4-maverick-17b-128e-instruct"
)

# Run prediction
response = predictor.predict(
    system_prompt="You are a helpful assistant specializing in image analysis.",
    user_prompt="Describe this image in detail.",
    temperature=0.7,
    max_output_tokens=500
)
```

Exceptions:
- `ValueError`: If the provider is not supported

---

### 3. `nvidia_predictor.py` - NVIDIA Implementation

Description: Concrete implementation for the NVIDIA API (OpenAI-compatible).

Features:
- ✅ OpenAI SDK: Uses OpenAI SDK with NVIDIA endpoint
- ✅ Retry Logic: Automatic retries with exponential backoff (tenacity)
- ✅ Environment Variables: API key loaded from `.env`

Main Class:
```python
class NVIDIAPredictor(LLMPredictor):
    def __init__(self, model: str):
        """Initialize with NVIDIA model name."""
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model
```

Retry Configuration:
- Attempts: Up to 3 times
- Backoff: Exponential (min=2s, max=5s, multiplier=1)
- Exceptions: Retry on any `Exception`

Supported Models (examples):
- `meta/llama-4-maverick-17b-128e-instruct`
- `meta/llama-4-scout-17b-16e-instruct`
- `mistralai/mistral-small-3.1-24b-instruct-2503`
- Any model available in the NVIDIA API

Usage Example:
```python
from src.core.llm_predictor import NVIDIAPredictor

predictor = NVIDIAPredictor("meta/llama-4-maverick-17b-128e-instruct")

response = predictor.predict(
    system_prompt="You are an expert in cultural analysis.",
    user_prompt="Analyze this carnival image.",
    temperature=0.0,
    max_output_tokens=1000
)
```

---

### 4. `__init__.py` - Public Interface

Description: Defines the module’s public exports.

Exports:
```python
__all__ = [
    'LLMPredictor',      # Abstract base class
    'LLMFactory',        # Factory for predictor creation
    'NVIDIAPredictor',   # NVIDIA implementation
]
```

Recommended Import:
```python
# Use the Factory (recommended)
from src.core.llm_predictor import LLMFactory

# Or import a specific implementation
from src.core.llm_predictor import NVIDIAPredictor
```

---

## 🚀 How to Use

### Method 1: Factory Pattern (Recommended)

```python
from src.core.llm_predictor import LLMFactory

# Create predictor
predictor = LLMFactory.get_predictor(
    llm_provider="nvidia",
    model_name="meta/llama-4-maverick-17b-128e-instruct"
)

# Run prediction
response = predictor.predict(
    system_prompt="You are a helpful assistant.",
    user_prompt="Hello! How are you?",
    temperature=0.7,
    max_output_tokens=200
)

print(response)
```

### Method 2: Direct Instantiation

```python
from src.core.llm_predictor import NVIDIAPredictor

# Create predictor directly
predictor = NVIDIAPredictor("meta/llama-4-maverick-17b-128e-instruct")

# Run prediction
response = predictor.predict(
    system_prompt="You are a helpful assistant.",
    user_prompt="Hello! How are you?",
    temperature=0.7,
    max_output_tokens=200
)

print(response)
```

### Using Images (Multimodal)

```python
from src.core.llm_predictor import LLMFactory

predictor = LLMFactory.get_predictor("nvidia", "meta/llama-4-maverick-17b-128e-instruct")

# User prompt with image tag
user_prompt = """
Analyze this image and describe what you see.

<image>
data:image/jpeg;base64,{base64_encoded_image}
</image>
"""

response = predictor.predict(
    system_prompt="You are an expert in image analysis.",
    user_prompt=user_prompt,
    temperature=0.0,
    max_output_tokens=1000
)
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file at the project root:

```bash
# NVIDIA API
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### Obtain API Keys

NVIDIA:
1. Go to: https://build.nvidia.com/
2. Create an account
3. Generate an API key
4. Add it to `.env`

---

## 🔧 Dependencies

```bash
pip install openai python-dotenv tenacity
```

Libraries Used:
- `openai` — OpenAI SDK (compatible with NVIDIA)
- `python-dotenv` — Environment variable loading
- `tenacity` — Retry logic with exponential backoff

---

## 🎯 Use Cases

### 1. Caption Generation
```python
from src.core.llm_predictor import LLMFactory

predictor = LLMFactory.get_predictor("nvidia", "meta/llama-4-maverick-17b-128e-instruct")

response = predictor.predict(
    system_prompt="You are an expert in generating image captions.",
    user_prompt="Generate a detailed caption for this carnival image: <image>...",
    temperature=0.0,
    max_output_tokens=500
)
```

### 2. Image Classification
```python
response = predictor.predict(
    system_prompt="You are an expert in image classification.",
    user_prompt="Classify this image into the following categories: ...",
    temperature=0.0,
    max_output_tokens=1000
)
```

### 3. Multi-turn Conversation
```python
# First message
response1 = predictor.predict(
    system_prompt="You are a helpful assistant.",
    user_prompt="What is the capital of Brazil?",
    temperature=0.7,
    max_output_tokens=100
)

# Second message (context maintained via system_prompt)
response2 = predictor.predict(
    system_prompt="You are a helpful assistant. Previous answer: Brasília",
    user_prompt="What is its population?",
    temperature=0.7,
    max_output_tokens=100
)
```

---

## 🧪 Testing

### Unit Test Example

```python
import unittest
from src.core.llm_predictor import LLMFactory

class TestLLMPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = LLMFactory.get_predictor("nvidia", "meta/llama-4-maverick-17b-128e-instruct")

    def test_basic_prediction(self):
        response = self.predictor.predict(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello!",
            temperature=0.0,
            max_output_tokens=50
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_deterministic_output(self):
        # Temperature 0.0 should be deterministic
        response1 = self.predictor.predict(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2?",
            temperature=0.0,
            max_output_tokens=50
        )
        response2 = self.predictor.predict(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2?",
            temperature=0.0,
            max_output_tokens=50
        )
        self.assertEqual(response1, response2)

if __name__ == '__main__':
    unittest.main()
```

---

## 🔍 Troubleshooting

### Error: "NVIDIA_API_KEY not found"
Solution: Ensure the `.env` file exists and contains the key
```bash
cat .env | grep NVIDIA_API_KEY
```

### Error: "Provider 'xyz' is not supported"
Solution: Currently only NVIDIA provider is supported:
```python
predictor = LLMFactory.get_predictor("nvidia", "model-name")  # ✅
predictor = LLMFactory.get_predictor("openai", "gpt-4")       # ❌ Not supported
```

### Error: Rate limit or timeout
Solution: NVIDIAPredictor already has automatic retries. Adjust if needed:
```python
# Edit nvidia_predictor.py
@retry(
    stop=stop_after_attempt(5),  # Increase attempts
    wait=wait_exponential(multiplier=2, min=4, max=10)  # Increase backoff
)
```

### Empty or truncated response
Solution: Increase `max_output_tokens`:
```python
response = predictor.predict(
    system_prompt="...",
    user_prompt="...",
    temperature=0.7,
    max_output_tokens=2000  # Increase limit
)
```

---

## 🔮 Roadmap

### Potential Future Features

- [ ] Streaming Support: Streaming responses
  ```python
  for chunk in predictor.predict_stream(...):
      print(chunk, end='', flush=True)
  ```

- [ ] Batch Processing: Multiple predictions in parallel
  ```python
  responses = predictor.predict_batch([
      {"system": "...", "user": "..."},
      {"system": "...", "user": "..."}
  ])
  ```

- [ ] Context Management: Automatic context handling
  ```python
  conversation = predictor.start_conversation()
  conversation.add_message("Hello!")
  response1 = conversation.get_response()
  conversation.add_message("How are you?")
  response2 = conversation.get_response()
  ```

---

## 📚 References

- NVIDIA API Documentation: https://build.nvidia.com/explore/discover
- Tenacity Documentation: https://tenacity.readthedocs.io/
- Python ABC Module: https://docs.python.org/3/library/abc.html
- Factory Pattern: https://refactoring.guru/design-patterns/factory-method

---

## 🤝 Contributing

To add a new LLM provider:

1. **Create predictor file**: `new_provider_predictor.py`
   ```python
   from src.core.llm_predictor.llm_predictor import LLMPredictor

   class NewProviderPredictor(LLMPredictor):
       def __init__(self, model: str):
           # Initialize client, API keys, etc.
           pass

       def predict(self, system_prompt, user_prompt, temperature, max_output_tokens):
           # Implementation with retry logic and error handling
           pass
   ```

2. **Add to Factory** (`llm_factory.py`):
   ```python
   from src.core.llm_predictor.new_provider_predictor import NewProviderPredictor

   # In get_predictor method:
   elif llm_provider == "newprovider":
       return NewProviderPredictor(model_name)
   ```

3. **Update exports** (`__init__.py`):
   ```python
   from src.core.llm_predictor.new_provider_predictor import NewProviderPredictor
   __all__ = ['LLMPredictor', 'LLMFactory', 'NVIDIAPredictor', 'NewProviderPredictor']
   ```

4. **Update README**: Add provider to supported list and examples

5. **Test thoroughly**: Ensure retry logic, error handling, and multimodal support work

---

## 📧 Contact

For questions about the LLM Predictor module, see the project’s main documentation or open an issue in the repository.
