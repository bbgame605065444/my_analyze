import unittest
from prompt_engineer import create_few_shot_prompt


class TestPromptEngineer(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.exemplars = [
            {
                'question': 'What is 2 + 3?',
                'answer': '5',
                'chain_of_thought': 'I need to add 2 and 3. 2 + 3 = 5.'
            },
            {
                'question': 'What is 10 - 4?',
                'answer': '6', 
                'chain_of_thought': 'I need to subtract 4 from 10. 10 - 4 = 6.'
            }
        ]
        self.new_question = 'What is 7 + 8?'
        
    def test_standard_prompting(self):
        """Test standard prompting without chain of thought"""
        prompt = create_few_shot_prompt(self.exemplars, self.new_question, use_cot=False)
        
        # Should contain questions and direct answers
        self.assertIn('Q: What is 2 + 3?', prompt)
        self.assertIn('A: 5', prompt)
        self.assertIn('Q: What is 10 - 4?', prompt)
        self.assertIn('A: 6', prompt)
        self.assertIn('Q: What is 7 + 8?', prompt)
        self.assertIn('A:', prompt)
        
        # Should NOT contain chain of thought
        self.assertNotIn('I need to add', prompt)
        self.assertNotIn('I need to subtract', prompt)
        
    def test_cot_prompting(self):
        """Test chain-of-thought prompting"""
        prompt = create_few_shot_prompt(self.exemplars, self.new_question, use_cot=True)
        
        # Should contain questions, reasoning, and answers
        self.assertIn('Q: What is 2 + 3?', prompt)
        self.assertIn('A: I need to add 2 and 3. 2 + 3 = 5. 5', prompt)
        self.assertIn('Q: What is 10 - 4?', prompt) 
        self.assertIn('A: I need to subtract 4 from 10. 10 - 4 = 6. 6', prompt)
        self.assertIn('Q: What is 7 + 8?', prompt)
        self.assertIn('A:', prompt)
        
    def test_empty_exemplars(self):
        """Test handling of empty exemplars list"""
        prompt = create_few_shot_prompt([], self.new_question, use_cot=False)
        self.assertIn('Q: What is 7 + 8?', prompt)
        self.assertIn('A:', prompt)
        
    def test_prompt_format_consistency(self):
        """Test that prompts follow consistent Q/A format"""
        for use_cot in [True, False]:
            prompt = create_few_shot_prompt(self.exemplars, self.new_question, use_cot)
            
            # Count Q: and A: occurrences
            q_count = prompt.count('Q:')
            a_count = prompt.count('A:')
            
            # Should have equal number of Q: and A: (including the final A:)
            self.assertEqual(q_count, a_count)
            
            # Should end with "A:"
            self.assertTrue(prompt.strip().endswith('A:'))
            
    def test_special_characters_handling(self):
        """Test handling of special characters in questions/answers"""
        special_exemplars = [
            {
                'question': 'What is "hello" + "world"?',
                'answer': 'helloworld',
                'chain_of_thought': 'Concatenating "hello" and "world" gives "helloworld".'
            }
        ]
        special_question = 'What is 5% of 100?'
        
        prompt = create_few_shot_prompt(special_exemplars, special_question, use_cot=True)
        self.assertIn('"hello"', prompt)
        self.assertIn('5%', prompt)


if __name__ == '__main__':
    unittest.main()