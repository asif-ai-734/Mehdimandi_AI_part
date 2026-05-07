"""
OpenAI integration service for chat completions and JSON analysis.
"""

import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import settings


class OpenAIService:
    """Service for interacting with OpenAI API."""
    
    SYSTEM_PROMPT = """You are a project-specific document assistant.
Answer the user only using the provided context from uploaded documents.
Do not use outside knowledge.
If the answer is not clearly supported by the context, say: "I could not find that information in the uploaded documents."
Cite sources with the bracketed labels provided in context, such as [S1].
Keep your answers concise and focused on the document content."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize OpenAI service.
        
        Args:
            api_key: OpenAI API key. Defaults to settings.openai_api_key
            model: Model name. Defaults to settings.openai_model
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is not configured")
        
        self.client = OpenAI(api_key=self.api_key)

    def refresh_from_settings(self) -> None:
        """Refresh this service if runtime OpenAI settings changed."""
        api_key = settings.openai_api_key
        model = settings.openai_model

        if self.api_key == api_key and self.model == model:
            return

        self.api_key = api_key
        self.model = model
        if not self.api_key:
            raise ValueError("OpenAI API key is not configured")

        self.client = OpenAI(api_key=self.api_key)
    
    def generate_response(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a response using OpenAI chat completion.
        
        Args:
            query: User's question
            context: Retrieved context from documents
            system_prompt: Optional custom system prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated response text
        """
        self.refresh_from_settings()
        system_prompt = system_prompt or self.SYSTEM_PROMPT
        
        user_message = f"""Context from uploaded documents:
{context}

User question: {query}

Please answer based only on the context provided above."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise RuntimeError(f"Error generating response from OpenAI: {str(e)}")
    
    def generate_response_with_history(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a response using OpenAI with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional custom system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated response text
        """
        self.refresh_from_settings()
        system_prompt = system_prompt or self.SYSTEM_PROMPT
        
        all_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise RuntimeError(f"Error generating response from OpenAI: {str(e)}")

    def generate_json(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2500
    ) -> Dict[str, Any]:
        """
        Generate and parse a JSON object from OpenAI.

        Messages may include multimodal content for drawing-page analysis.
        """
        self.refresh_from_settings()
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        request = {
            "model": model or self.model,
            "messages": all_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        try:
            response = self.client.chat.completions.create(**request)
        except Exception:
            request.pop("response_format", None)
            response = self.client.chat.completions.create(**request)

        content = response.choices[0].message.content or "{}"
        return self._parse_json_object(content)

    @staticmethod
    def _parse_json_object(content: str) -> Dict[str, Any]:
        """Parse a JSON object, tolerating fenced markdown around the JSON."""
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {"value": parsed}
    
    def extract_sources_from_response(
        self,
        response: str,
        source_filenames: List[str]
    ) -> List[str]:
        """
        Extract cited source filenames from the response.
        
        Args:
            response: Generated response text
            source_filenames: List of available source filenames
        
        Returns:
            List of cited filenames
        """
        cited_sources = []
        for source in source_filenames:
            label = source.split(":", 1)[0].strip()
            if f"[{label}]" in response or source.lower() in response.lower():
                cited_sources.append(source)
        return cited_sources or source_filenames


# Global OpenAI service instance
_openai_service = None


def get_openai_service() -> OpenAIService:
    """Get or create the global OpenAI service instance."""
    global _openai_service
    if (
        _openai_service is None
        or _openai_service.api_key != settings.openai_api_key
        or _openai_service.model != settings.openai_model
    ):
        _openai_service = OpenAIService()
    return _openai_service


def reset_openai_service() -> None:
    """Force the next OpenAI service lookup to use current runtime settings."""
    global _openai_service
    _openai_service = None


def generate_chat_response(query: str, context: str) -> str:
    """Convenience function to generate a chat response."""
    service = get_openai_service()
    return service.generate_response(query, context)
