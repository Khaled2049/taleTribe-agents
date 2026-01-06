"""LLM provider abstraction for supporting multiple AI backends."""
import os
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import httpx


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """
        Generate content from a prompt.

        Args:
            prompt: The input prompt

        Returns:
            Generated text content
        """
        pass

    @abstractmethod
    async def generate_structured_content(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> List[str]:
        """
        Generate structured content with a JSON schema constraint.

        Args:
            system_prompt: System-level instructions for the AI
            user_prompt: User-level prompt with the actual task
            response_schema: JSON schema defining the expected response structure

        Returns:
            List of strings (parsed from the structured JSON response)
        """
        pass


class GoogleAIStudioProvider(LLMProvider):
    """Google AI Studio (Gemini) provider using REST API with API key."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Google AI Studio provider.

        Args:
            api_key: Google AI Studio API key
            model_name: Model name to use (default: gemini-2.0-flash-exp for free tier)
                        Use gemini-2.0-flash-exp for free tier, gemini-1.5-flash for paid
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def generate_content(self, prompt: str) -> str:
        """Generate content using Google AI Studio API."""
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        
        try:
            response = httpx.post(
                url,
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                },
                params={"key": self.api_key},
                timeout=300.0,
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract text from response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            
            raise ValueError(f"Unexpected response structure: {result}")
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to Google AI Studio API: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Google AI Studio API error: {e.response.status_code} - {e.response.text}")

    async def generate_structured_content(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> List[str]:
        """Generate structured content using Google AI Studio API."""
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        
        
        full_prompt = (
            f"{system_prompt}\n\n"
            f"{user_prompt}\n\n"
            "IMPORTANT: Output ONLY the JSON array."
        )

        try:
            response = httpx.post(
                url,
                json={
                    "contents": [{
                        "parts": [{"text": full_prompt}]
                    }],
                    "generationConfig": {
                        "responseMimeType": "application/json",                        
                        "responseSchema": response_schema 
                    }
                },
                params={"key": self.api_key},
                timeout=300.0,
            )
            response.raise_for_status()
            result = response.json()
            
            
            print(f"[GOOGLE_AI_STUDIO_PROVIDER] Google AI Studio API response: {result}")
            if "candidates" in result and result["candidates"]:
                candidate = result["candidates"][0]
                content_parts = candidate.get("content", {}).get("parts", [])
                
                if content_parts and "text" in content_parts[0]:
                    response_text = content_parts[0]["text"].strip()
                    
                    # Clean up any Markdown code blocks (Gemini sometimes adds them despite settings)
                    if response_text.startswith("```"):
                        response_text = response_text.replace("```json", "").replace("```", "")

                    try:
                        json_data = json.loads(response_text)
                        
                        # Case A: It's a clean list ["a", "b"]
                        if isinstance(json_data, list):
                            return [str(item) for item in json_data]
                            
                        # Case B: It's a dict (sometimes Gemini wraps arrays in objects)
                        # e.g. { "items": ["a", "b"] } or { "suggestions": ["a", "b"] }
                        if isinstance(json_data, dict):
                            # Look for the first list value found in the dict
                            for key, value in json_data.items():
                                if isinstance(value, list):
                                    return [str(item) for item in value]
                            
                            # Fallback: maybe the dict itself is the item?
                            return []

                        return []
                        
                    except json.JSONDecodeError:
                        print(f"[ERROR] Failed to decode JSON: {response_text}")
                        return []
            
            return []

        except Exception as e:
            print(f"[ERROR] API Call failed: {e}")
            return []

class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2"):
        """
        Initialize Ollama provider.

        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
            model_name: Model name to use (default: llama3.2)
                      Common models: llama3.2, mistral, phi3, gemma2, etc.
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_url = f"{self.base_url}/api/generate"

    def generate_content(self, prompt: str) -> str:
        """Generate content using Ollama."""
        try:
            response = httpx.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=300.0,  # 5 minutes timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}")

    async def generate_structured_content(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> List[str]:
        """Generate structured content using Ollama with JSON parsing."""
        # Since Ollama doesn't natively support JSON schema, enhance the prompt
        # to explicitly request JSON format
        schema_description = json.dumps(response_schema, indent=2)
        enhanced_prompt = f"""{system_prompt}

{user_prompt}

IMPORTANT: You MUST respond with ONLY a valid JSON array that matches this schema:
{schema_description}

Do not include any text before or after the JSON array. Return ONLY the JSON array."""

        try:
            response = httpx.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": enhanced_prompt,
                    "stream": False,
                },
                timeout=300.0,  # 5 minutes timeout
            )
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "").strip()
            
            # Try to extract JSON from the response (may have extra text)
            # Look for JSON array pattern
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            # Parse JSON response
            try:
                json_data = json.loads(response_text)
                # Extract the array from the response
                if isinstance(json_data, list):
                    return json_data
                elif isinstance(json_data, dict) and "items" in json_data:
                    return json_data["items"]
                else:
                    raise ValueError(f"Unexpected JSON structure: {json_data}")
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON response from Ollama: {e}. "
                    f"Response text: {response_text[:200]}..."
                )
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}")


class MockProvider(LLMProvider):
    """Mock provider for testing without any AI calls."""

    def generate_content(self, prompt: str) -> str:
        """Generate mock content."""
        # Simple mock that returns formatted responses based on prompt content
        if "character" in prompt.lower():
            return """1. **Aria Blackwood** - A mysterious scholar with a hidden past, seeking ancient knowledge. Key traits: Intelligent, secretive, determined. Backstory: Former member of a secret organization, now on the run. Motivations: To uncover the truth about her family's disappearance.

2. **Marcus Thorne** - A skilled warrior with a code of honor. Key traits: Brave, loyal, conflicted. Backstory: Raised in a military academy, left after questioning orders. Motivations: To protect the innocent and find redemption.

3. **Luna Starweaver** - A magical healer with a connection to nature. Key traits: Compassionate, intuitive, powerful. Backstory: Discovered her powers during a childhood illness. Motivations: To heal the world and restore balance."""
        
        elif "plot" in prompt.lower():
            return """1. **The Hidden Prophecy** - An ancient prophecy reveals that the main character must make a difficult choice between saving their loved one or saving the world. This creates internal conflict and drives the story forward.

2. **The Betrayal** - A trusted ally is revealed to be working for the antagonist, creating tension and forcing the protagonist to question everyone around them.

3. **The Discovery** - The protagonist discovers that their greatest enemy is actually trying to prevent a greater evil, forcing them to reconsider their entire mission."""
        
        elif "place" in prompt.lower():
            return """1. **The Whispering Woods** - A mystical forest where the trees seem to speak secrets. The atmosphere is eerie yet beautiful, with ancient magic lingering in the air. Key features include glowing mushrooms and a hidden clearing where important events occur.

2. **The Crystal Caves** - Underground caverns filled with luminescent crystals that store magical energy. These caves serve as a sanctuary and a source of power for the characters.

3. **The Forgotten Library** - An abandoned library containing lost knowledge and dangerous secrets. The shelves stretch endlessly, and some books are said to be alive."""
        
        elif "theme" in prompt.lower():
            return """1. **Sacrifice and Redemption** - Exploring how characters must give up something important to achieve their goals and find redemption for past mistakes.

2. **The Nature of Power** - Examining how power corrupts and how different characters handle authority and responsibility.

3. **Identity and Self-Discovery** - Characters questioning who they are and discovering their true purpose in the world."""
        
        else:
            # Generic response
            return f"""This is a mock response for testing purposes.

Prompt received: {prompt[:100]}...

[Mock content would be generated here in a real scenario. This allows you to test the API flow without incurring costs or requiring an AI model.]"""

    async def generate_structured_content(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> List[str]:
        """Generate mock structured content for testing."""
        # Return mock suggestions based on context
        # Check if the prompt mentions next lines or continuation
        if "next line" in user_prompt.lower() or "continuation" in user_prompt.lower() or "[INSERTION_POINT]" in user_prompt:
            return [
                "The morning light filtered through the curtains, casting long shadows across the room.",
                "She paused, considering her next words carefully before speaking.",
                "A sense of unease settled over him as he realized what was about to happen.",
            ]
        else:
            # Generic mock suggestions
            return [
                "Mock suggestion 1 for structured content testing.",
                "Mock suggestion 2 for structured content testing.",
                "Mock suggestion 3 for structured content testing.",
            ]


def get_llm_provider(project_id: Optional[str] = None, location: str = "us-central1") -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider based on environment variables.

    Environment variables:
    - GOOGLE_AI_STUDIO_API_KEY: If set, use Google AI Studio API (REST API with API key) - DEFAULT in production
    - USE_OLLAMA: If set to "true", use Ollama for local development
    - OLLAMA_BASE_URL: Ollama API base URL (default: http://localhost:11434)
    - OLLAMA_MODEL: Model name to use (default: phi4-mini)
    - USE_MOCK: If set to "true", use mock provider (no AI calls, for testing)
    - GOOGLE_AI_STUDIO_MODEL: Model name for Google AI Studio (default: gemini-2.0-flash-exp for free tier)

    Args:
        project_id: GCP project ID (required for Firestore access, not used by LLM provider)
        location: GCP location (not used, kept for backward compatibility)

    Returns:
        LLMProvider instance
    """
    use_mock = os.getenv("USE_MOCK", "").lower() == "true"
    use_ollama = os.getenv("USE_OLLAMA", "").lower() == "true"
    google_ai_studio_api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")

    if use_mock:
        print("Using Mock LLM Provider for testing.")
        return MockProvider()
    
    if use_ollama:
        print("Using Ollama LLM Provider.")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "phi4-mini")
        return OllamaProvider(base_url=base_url, model_name=model_name)
    
    # Prefer Google AI Studio API if API key is provided
    if google_ai_studio_api_key:
        print("Using Google AI Studio API Provider.")
        model_name = os.getenv("GOOGLE_AI_STUDIO_MODEL", "gemini-2.0-flash-exp")
        return GoogleAIStudioProvider(api_key=google_ai_studio_api_key, model_name=model_name)
    
    # Default to ollama
    print("Using Ollama Provider.")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "phi4-mini")
    return OllamaProvider(base_url=base_url, model_name=model_name)

