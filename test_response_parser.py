import unittest
from response_parser import parse_response, evaluate_answer


class TestResponseParser(unittest.TestCase):
    
    def test_parse_numeric_answer(self):
        """Test parsing numeric answers from responses"""
        responses = [
            "The calculation is 2 + 3 = 5. The answer is 5.",
            "Let me work through this step by step. First, 2 + 3 = 5. So the final answer is 5.",
            "2 + 3 = 5",
            "The answer is 42.",
            "After working through the problem, I get 123.45 as the result."
        ]
        
        expected = ["5", "5", "5", "42", "123.45"]
        
        for response, expected_answer in zip(responses, expected):
            result = parse_response(response)
            self.assertEqual(result, expected_answer)
            
    def test_parse_text_answer(self):
        """Test parsing text answers from responses"""
        responses = [
            "The answer is yes.",
            "After considering all options, the answer is no.",
            "The capital of France is Paris.",
            "Looking at the evidence, I conclude that the answer is maybe."
        ]
        
        expected = ["yes", "no", "Paris", "maybe"]
        
        for response, expected_answer in zip(responses, expected):
            result = parse_response(response)
            self.assertEqual(result, expected_answer)
            
    def test_parse_last_number_fallback(self):
        """Test fallback to last number when 'answer is' not found"""
        responses = [
            "First I calculate 10 + 5 = 15, then multiply by 2 to get 30.",
            "The calculation yields 42 after several steps involving 7 and 6.",
            "Through step-by-step reasoning: 100 - 25 = 75."
        ]
        
        expected = ["30", "42", "75"]
        
        for response, expected_answer in zip(responses, expected):
            result = parse_response(response)
            self.assertEqual(result, expected_answer)
            
    def test_parse_no_answer_found(self):
        """Test handling when no clear answer is found"""
        responses = [
            "This is a complex problem without a clear solution.",
            "I cannot determine the answer from the given information.",
            ""
        ]
        
        for response in responses:
            result = parse_response(response)
            self.assertEqual(result, "")
            
    def test_evaluate_exact_match(self):
        """Test exact answer matching"""
        test_cases = [
            ("42", "42", True),
            ("yes", "yes", True),
            ("Paris", "Paris", True),
            ("42", "43", False),
            ("yes", "no", False)
        ]
        
        for predicted, true_answer, expected in test_cases:
            result = evaluate_answer(predicted, true_answer)
            self.assertEqual(result, expected)
            
    def test_evaluate_case_insensitive(self):
        """Test case-insensitive answer matching"""
        test_cases = [
            ("YES", "yes", True),
            ("Paris", "paris", True),
            ("NO", "no", True),
            ("Maybe", "MAYBE", True)
        ]
        
        for predicted, true_answer, expected in test_cases:
            result = evaluate_answer(predicted, true_answer)
            self.assertEqual(result, expected)
            
    def test_evaluate_whitespace_normalization(self):
        """Test whitespace normalization in answer matching"""
        test_cases = [
            ("  42  ", "42", True),
            ("42", "  42  ", True),
            (" yes ", "yes", True),
            ("New York", "New  York", True)
        ]
        
        for predicted, true_answer, expected in test_cases:
            result = evaluate_answer(predicted, true_answer)
            self.assertEqual(result, expected)
            
    def test_evaluate_punctuation_handling(self):
        """Test punctuation handling in answer matching"""
        test_cases = [
            ("42.", "42", True),
            ("yes!", "yes", True),
            ("Paris,", "Paris", True),
            ("42", "42.", True)
        ]
        
        for predicted, true_answer, expected in test_cases:
            result = evaluate_answer(predicted, true_answer)
            self.assertEqual(result, expected)
            
    def test_parse_complex_cot_response(self):
        """Test parsing responses with complex chain-of-thought reasoning"""
        response = """
        Let me think through this step by step.
        
        First, I need to understand what the question is asking.
        The question wants me to find the sum of 15 and 27.
        
        Step 1: 15 + 27
        Step 2: I can break this down as 15 + 20 + 7 = 35 + 7 = 42
        
        Therefore, the answer is 42.
        """
        
        result = parse_response(response)
        self.assertEqual(result, "42")
        
    def test_parse_multiple_numbers_in_response(self):
        """Test parsing when response contains multiple numbers"""
        response = "I calculated 10 + 5 = 15, then 15 * 2 = 30. The final answer is 30."
        result = parse_response(response)
        self.assertEqual(result, "30")  # Should get the final answer, not intermediate calculations


if __name__ == '__main__':
    unittest.main()