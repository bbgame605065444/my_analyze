"""
Enhanced LLM Interface for CoT-RAG Framework
Extends the existing model_interface.py with CoT-RAG specific functionality.
"""

from typing import Dict, Any, Optional, List
from model_interface import get_model_response
from model_config import ConfigManager, ModelConfig

class LLMInterface:
    """
    Enhanced LLM interface that wraps the existing model_interface.py
    with CoT-RAG specific functionality and error handling.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize LLM interface.
        
        Args:
            config: Model configuration (uses ConfigManager if None)
        """
        if config is None:
            config_manager = ConfigManager()
            self.config = config_manager.get_config()
        else:
            self.config = config
        
        # Track usage statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens_used': 0,
            'calls_by_type': {}
        }
    
    def query(self, prompt: str, model_name: str = "default", 
              retry_count: int = 3, **kwargs) -> str:
        """
        Query the LLM with retry logic and error handling.
        
        Args:
            prompt: The input prompt
            model_name: Model identifier for tracking (not used in actual call)
            retry_count: Number of retries on failure
            **kwargs: Additional parameters for the model
            
        Returns:
            str: Model response
            
        Raises:
            Exception: If all retries fail
        """
        # Update statistics
        self.stats['total_calls'] += 1
        self.stats['calls_by_type'][model_name] = self.stats['calls_by_type'].get(model_name, 0) + 1
        
        # Prepare model parameters
        model_params = {
            'temperature': kwargs.get('temperature', self.config.temperature),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'top_p': kwargs.get('top_p', self.config.top_p)
        }
        
        last_error = None
        
        for attempt in range(retry_count):
            try:
                # Call the existing model interface
                response = get_model_response(prompt, self.config)
                
                # Update success statistics
                self.stats['successful_calls'] += 1
                
                # Estimate token usage (rough approximation)
                estimated_tokens = len(prompt.split()) + len(response.split())
                self.stats['total_tokens_used'] += estimated_tokens
                
                return response
                
            except Exception as e:
                last_error = e
                print(f"LLM call attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < retry_count - 1:
                    import time
                    time.sleep(self.config.base_delay * (2 ** attempt))  # Exponential backoff
                    continue
        
        # All retries failed
        self.stats['failed_calls'] += 1
        raise Exception(f"LLM query failed after {retry_count} attempts. Last error: {last_error}")
    
    def query_structured(self, prompt: str, expected_format: str = "json",
                        model_name: str = "structured", **kwargs) -> Dict[str, Any]:
        """
        Query LLM for structured output (JSON).
        
        Args:
            prompt: Input prompt
            expected_format: Expected output format
            model_name: Model identifier
            **kwargs: Additional parameters
            
        Returns:
            Dict: Parsed structured response
        """
        # Add format instruction to prompt
        format_instruction = f"\n\nIMPORTANT: Respond with valid {expected_format.upper()} format only."
        full_prompt = prompt + format_instruction
        
        response = self.query(full_prompt, model_name, **kwargs)
        
        # Try to parse JSON response
        if expected_format.lower() == "json":
            return self._parse_json_response(response)
        else:
            return {"raw_response": response}
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with error handling."""
        import json
        import re
        
        try:
            # Try direct parsing first
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON-like structure
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If all parsing fails, return error structure
        return {
            "error": "Failed to parse JSON response",
            "raw_response": response[:500] + "..." if len(response) > 500 else response
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        return {
            **self.stats,
            'success_rate': (self.stats['successful_calls'] / max(self.stats['total_calls'], 1)) * 100,
            'config': {
                'model_type': self.config.model_type,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens
            }
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens_used': 0,
            'calls_by_type': {}
        }

# Convenience functions for backward compatibility
def query_llm(prompt: str, config: Optional[ModelConfig] = None, **kwargs) -> str:
    """
    Convenience function for single LLM queries.
    
    Args:
        prompt: Input prompt
        config: Model configuration
        **kwargs: Additional parameters
        
    Returns:
        str: Model response
    """
    interface = LLMInterface(config)
    return interface.query(prompt, **kwargs)

def query_llm_json(prompt: str, config: Optional[ModelConfig] = None, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for structured JSON queries.
    
    Args:
        prompt: Input prompt
        config: Model configuration
        **kwargs: Additional parameters
        
    Returns:
        Dict: Parsed JSON response
    """
    interface = LLMInterface(config)
    return interface.query_structured(prompt, "json", **kwargs)

# Global interface instance for reuse
_global_interface = None

def get_global_llm_interface() -> LLMInterface:
    """Get or create global LLM interface instance."""
    global _global_interface
    if _global_interface is None:
        _global_interface = LLMInterface()
    return _global_interface