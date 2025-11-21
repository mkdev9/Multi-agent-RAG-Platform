"""LLM service for interacting with language model providers."""
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json

import openai
import httpx
from openai import AsyncOpenAI
import boto3

from ..config import settings
from ..core.exceptions import LLMError, ConfigurationError
from ..core.logging import BusinessLogger

logger = BusinessLogger()


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    async def generate_embedding(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """Generate text embedding."""
        pass
    
    @abstractmethod
    async def batch_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate batch embeddings."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self):
        if not settings.llm.openai_api_key:
            raise ConfigurationError("OpenAI API key not configured")
        
        self.client = AsyncOpenAI(api_key=settings.llm.openai_api_key)
        self.model = settings.llm.openai_model
        self.embedding_model = settings.llm.openai_embedding_model
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "provider": "openai"
            }
        
        except Exception as e:
            raise LLMError(f"OpenAI API error: {str(e)}")
    
    async def generate_embedding(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """Generate embedding using OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                **kwargs
            )
            return response.data[0].embedding
        
        except Exception as e:
            raise LLMError(f"OpenAI embedding error: {str(e)}")
    
    async def batch_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate batch embeddings using OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                **kwargs
            )
            return [item.embedding for item in response.data]
        
        except Exception as e:
            raise LLMError(f"OpenAI batch embedding error: {str(e)}")


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider implementation."""
    
    def __init__(self):
        if not all([
            settings.llm.azure_openai_key,
            settings.llm.azure_openai_endpoint
        ]):
            raise ConfigurationError("Azure OpenAI configuration incomplete")
        
        self.client = AsyncOpenAI(
            api_key=settings.llm.azure_openai_key,
            azure_endpoint=settings.llm.azure_openai_endpoint,
            api_version=settings.llm.azure_openai_version,
        )
        self.model = settings.llm.openai_model
        self.embedding_model = settings.llm.openai_embedding_model
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Azure OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "provider": "azure_openai"
            }
        
        except Exception as e:
            raise LLMError(f"Azure OpenAI API error: {str(e)}")
    
    async def generate_embedding(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """Generate embedding using Azure OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                **kwargs
            )
            return response.data[0].embedding
        
        except Exception as e:
            raise LLMError(f"Azure OpenAI embedding error: {str(e)}")
    
    async def batch_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate batch embeddings using Azure OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                **kwargs
            )
            return [item.embedding for item in response.data]
        
        except Exception as e:
            raise LLMError(f"Azure OpenAI batch embedding error: {str(e)}")


class BedrockProvider(LLMProvider):
    """AWS Bedrock provider implementation."""
    
    def __init__(self):
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=settings.llm.aws_region,
            aws_access_key_id=settings.llm.aws_access_key_id,
            aws_secret_access_key=settings.llm.aws_secret_access_key,
        )
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Default model
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using AWS Bedrock."""
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            })
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.invoke_model(
                    modelId=self.model_id,
                    body=body
                )
            )
            
            response_body = json.loads(response['body'].read())
            
            return {
                "content": response_body['content'][0]['text'],
                "model": self.model_id,
                "usage": {
                    "prompt_tokens": response_body.get('usage', {}).get('input_tokens', 0),
                    "completion_tokens": response_body.get('usage', {}).get('output_tokens', 0),
                    "total_tokens": response_body.get('usage', {}).get('input_tokens', 0) + 
                                  response_body.get('usage', {}).get('output_tokens', 0),
                },
                "provider": "aws_bedrock"
            }
        
        except Exception as e:
            raise LLMError(f"AWS Bedrock error: {str(e)}")
    
    async def generate_embedding(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """Generate embedding using AWS Bedrock."""
        # Note: Bedrock embedding models vary by region and availability
        # This is a placeholder implementation
        raise LLMError("Bedrock embedding not implemented in this example")
    
    async def batch_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate batch embeddings using AWS Bedrock."""
        raise LLMError("Bedrock batch embedding not implemented in this example")


class HuggingFaceProvider(LLMProvider):
    """Hugging Face provider implementation."""
    
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.model = "microsoft/DialoGPT-medium"  # Default model
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Hugging Face."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}{self.model}",
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                            **kwargs
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    raise LLMError(f"Hugging Face API error: {response.text}")
                
                result = response.json()
                
                return {
                    "content": result[0].get("generated_text", ""),
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": len(prompt.split()),  # Approximate
                        "completion_tokens": len(result[0].get("generated_text", "").split()),
                        "total_tokens": len(prompt.split()) + len(result[0].get("generated_text", "").split()),
                    },
                    "provider": "huggingface"
                }
        
        except Exception as e:
            raise LLMError(f"Hugging Face error: {str(e)}")
    
    async def generate_embedding(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """Generate embedding using Hugging Face."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}{self.embedding_model}",
                    json={"inputs": text},
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    raise LLMError(f"Hugging Face embedding error: {response.text}")
                
                return response.json()
        
        except Exception as e:
            raise LLMError(f"Hugging Face embedding error: {str(e)}")
    
    async def batch_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate batch embeddings using Hugging Face."""
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text, **kwargs)
            embeddings.append(embedding)
        return embeddings


class LLMService:
    """Main LLM service that manages multiple providers."""
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.default_provider = settings.llm.default_provider
        
        # Initialize available providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers."""
        try:
            if settings.llm.openai_api_key:
                self.providers["openai"] = OpenAIProvider()
        except Exception as e:
            print(f"Failed to initialize OpenAI provider: {e}")
        
        try:
            if settings.llm.azure_openai_key:
                self.providers["azure_openai"] = AzureOpenAIProvider()
        except Exception as e:
            print(f"Failed to initialize Azure OpenAI provider: {e}")
        
        try:
            if settings.llm.aws_access_key_id:
                self.providers["aws_bedrock"] = BedrockProvider()
        except Exception as e:
            print(f"Failed to initialize AWS Bedrock provider: {e}")
        
        try:
            self.providers["huggingface"] = HuggingFaceProvider()
        except Exception as e:
            print(f"Failed to initialize Hugging Face provider: {e}")
        
        if not self.providers:
            raise ConfigurationError("No LLM providers configured")
    
    def get_provider(self, provider_name: Optional[str] = None) -> LLMProvider:
        """Get LLM provider by name."""
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            raise ConfigurationError(f"Provider {provider_name} not available")
        
        return self.providers[provider_name]
    
    async def generate_response(
        self,
        prompt: str,
        provider: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using specified provider."""
        llm_provider = self.get_provider(provider)
        return await llm_provider.generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    async def generate_embedding(
        self,
        text: str,
        provider: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """Generate text embedding using specified provider."""
        llm_provider = self.get_provider(provider)
        return await llm_provider.generate_embedding(text=text, **kwargs)
    
    async def batch_embeddings(
        self,
        texts: List[str],
        provider: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """Generate batch embeddings using specified provider."""
        llm_provider = self.get_provider(provider)
        return await llm_provider.batch_embeddings(texts=texts, **kwargs)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())


# Global LLM service instance
llm_service = LLMService()