import unittest
from unittest.mock import patch, Mock
from model_interface import get_model_response
import time


class TestModelInterface(unittest.TestCase):
    
    def test_successful_response(self):
        """Test successful model response"""
        with patch('model_interface.genai') as mock_genai:
            # Mock successful response
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "The answer is 42."
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            result = get_model_response("What is 6 * 7?")
            self.assertEqual(result, "The answer is 42.")
            
    def test_api_retry_logic(self):
        """Test retry logic on API failures"""
        with patch('model_interface.genai') as mock_genai:
            mock_model = Mock()
            
            # First call fails, second succeeds
            mock_response = Mock()
            mock_response.text = "Success"
            mock_model.generate_content.side_effect = [
                Exception("API Error"),
                mock_response
            ]
            mock_genai.GenerativeModel.return_value = mock_model
            
            with patch('time.sleep'):  # Speed up test
                result = get_model_response("Test prompt")
                
            self.assertEqual(result, "Success")
            self.assertEqual(mock_model.generate_content.call_count, 2)
            
    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded"""
        with patch('model_interface.genai') as mock_genai:
            mock_model = Mock()
            mock_model.generate_content.side_effect = Exception("Persistent API Error")
            mock_genai.GenerativeModel.return_value = mock_model
            
            with patch('time.sleep'):  # Speed up test
                with self.assertRaises(Exception):
                    get_model_response("Test prompt")
                    
    def test_empty_prompt_handling(self):
        """Test handling of empty prompts"""
        with patch('model_interface.genai') as mock_genai:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = ""
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            result = get_model_response("")
            self.assertEqual(result, "")
            
    def test_exponential_backoff(self):
        """Test that retry delays follow exponential backoff"""
        with patch('model_interface.genai') as mock_genai:
            mock_model = Mock()
            mock_model.generate_content.side_effect = [
                Exception("Error 1"),
                Exception("Error 2"), 
                Exception("Error 3")
            ]
            mock_genai.GenerativeModel.return_value = mock_model
            
            sleep_times = []
            def mock_sleep(duration):
                sleep_times.append(duration)
                
            with patch('time.sleep', side_effect=mock_sleep):
                try:
                    get_model_response("Test prompt")
                except:
                    pass
                    
            # Verify exponential backoff (1, 2, 4 seconds)
            expected_times = [1, 2, 4]
            self.assertEqual(sleep_times, expected_times)
            
    def test_long_prompt_handling(self):
        """Test handling of very long prompts"""
        long_prompt = "A" * 10000  # Very long prompt
        
        with patch('model_interface.genai') as mock_genai:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "Response to long prompt"
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            result = get_model_response(long_prompt)
            self.assertEqual(result, "Response to long prompt")


if __name__ == '__main__':
    unittest.main()