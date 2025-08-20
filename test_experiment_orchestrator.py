import unittest
from unittest.mock import patch, Mock
from experiment_orchestrator import run_experiment


class TestExperimentOrchestrator(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.mock_dataset = [
            {
                'question': 'What is 2 + 3?',
                'answer': '5',
                'chain_of_thought': 'I need to add 2 and 3. 2 + 3 = 5.'
            },
            {
                'question': 'What is 10 - 4?',
                'answer': '6',
                'chain_of_thought': 'I need to subtract 4 from 10. 10 - 4 = 6.'
            },
            {
                'question': 'What is 7 * 8?',
                'answer': '56',
                'chain_of_thought': 'I need to multiply 7 and 8. 7 * 8 = 56.'
            },
            {
                'question': 'What is 20 / 4?',
                'answer': '5',
                'chain_of_thought': 'I need to divide 20 by 4. 20 / 4 = 5.'
            }
        ]
        
    @patch('experiment_orchestrator.load_dataset')
    @patch('experiment_orchestrator.create_few_shot_prompt')
    @patch('experiment_orchestrator.get_model_response')
    @patch('experiment_orchestrator.parse_response')
    @patch('experiment_orchestrator.evaluate_answer')
    def test_perfect_accuracy(self, mock_evaluate, mock_parse, mock_response, 
                             mock_prompt, mock_load):
        """Test experiment with 100% accuracy"""
        # Setup mocks
        mock_load.return_value = self.mock_dataset
        mock_prompt.return_value = "Mock prompt"
        mock_response.return_value = "Mock response"
        mock_parse.return_value = "correct_answer"
        mock_evaluate.return_value = True  # All answers correct
        
        accuracy = run_experiment('gsm8k', use_cot=True)
        
        # Should achieve 100% accuracy (2 test questions out of 4 total)
        self.assertEqual(accuracy, 1.0)
        
        # Verify function calls
        mock_load.assert_called_once_with('gsm8k')
        self.assertEqual(mock_evaluate.call_count, 2)  # 2 test questions
        
    @patch('experiment_orchestrator.load_dataset')
    @patch('experiment_orchestrator.create_few_shot_prompt')
    @patch('experiment_orchestrator.get_model_response')
    @patch('experiment_orchestrator.parse_response')
    @patch('experiment_orchestrator.evaluate_answer')
    def test_partial_accuracy(self, mock_evaluate, mock_parse, mock_response,
                             mock_prompt, mock_load):
        """Test experiment with partial accuracy"""
        # Setup mocks
        mock_load.return_value = self.mock_dataset
        mock_prompt.return_value = "Mock prompt"
        mock_response.return_value = "Mock response"
        mock_parse.return_value = "answer"
        mock_evaluate.side_effect = [True, False]  # 50% accuracy
        
        accuracy = run_experiment('gsm8k', use_cot=False)
        
        # Should achieve 50% accuracy
        self.assertEqual(accuracy, 0.5)
        
    @patch('experiment_orchestrator.load_dataset')
    @patch('experiment_orchestrator.create_few_shot_prompt')
    @patch('experiment_orchestrator.get_model_response')
    @patch('experiment_orchestrator.parse_response')
    @patch('experiment_orchestrator.evaluate_answer')
    def test_zero_accuracy(self, mock_evaluate, mock_parse, mock_response,
                          mock_prompt, mock_load):
        """Test experiment with 0% accuracy"""
        # Setup mocks
        mock_load.return_value = self.mock_dataset
        mock_prompt.return_value = "Mock prompt"
        mock_response.return_value = "Mock response"
        mock_parse.return_value = "wrong_answer"
        mock_evaluate.return_value = False  # All answers wrong
        
        accuracy = run_experiment('csqa', use_cot=True)
        
        # Should achieve 0% accuracy
        self.assertEqual(accuracy, 0.0)
        
    @patch('experiment_orchestrator.load_dataset')
    def test_few_shot_exemplar_selection(self, mock_load):
        """Test that first 2 examples are used as few-shot exemplars"""
        mock_load.return_value = self.mock_dataset
        
        with patch('experiment_orchestrator.create_few_shot_prompt') as mock_prompt:
            with patch('experiment_orchestrator.get_model_response'):
                with patch('experiment_orchestrator.parse_response'):
                    with patch('experiment_orchestrator.evaluate_answer'):
                        run_experiment('gsm8k', use_cot=True)
        
        # Check that create_few_shot_prompt was called with first 2 examples
        expected_exemplars = self.mock_dataset[:2]
        calls = mock_prompt.call_args_list
        
        # Should be called twice (once for each test question)
        self.assertEqual(len(calls), 2)
        
        # Each call should use the same exemplars (first 2 examples)
        for call in calls:
            exemplars = call[0][0]  # First argument
            self.assertEqual(exemplars, expected_exemplars)
            
    @patch('experiment_orchestrator.load_dataset')
    @patch('experiment_orchestrator.create_few_shot_prompt')
    @patch('experiment_orchestrator.get_model_response')
    @patch('experiment_orchestrator.parse_response')
    @patch('experiment_orchestrator.evaluate_answer')
    def test_cot_flag_propagation(self, mock_evaluate, mock_parse, mock_response,
                                 mock_prompt, mock_load):
        """Test that use_cot flag is properly propagated"""
        mock_load.return_value = self.mock_dataset
        mock_evaluate.return_value = True
        
        # Test with CoT enabled
        run_experiment('gsm8k', use_cot=True)
        
        # Check that create_few_shot_prompt was called with use_cot=True
        calls = mock_prompt.call_args_list
        for call in calls:
            use_cot_arg = call[0][2]  # Third argument
            self.assertTrue(use_cot_arg)
            
        # Reset mock and test with CoT disabled
        mock_prompt.reset_mock()
        run_experiment('gsm8k', use_cot=False)
        
        calls = mock_prompt.call_args_list
        for call in calls:
            use_cot_arg = call[0][2]  # Third argument
            self.assertFalse(use_cot_arg)
            
    @patch('experiment_orchestrator.load_dataset')
    def test_empty_dataset_handling(self, mock_load):
        """Test handling of empty datasets"""
        mock_load.return_value = []
        
        accuracy = run_experiment('empty_task', use_cot=True)
        
        # Should return 0.0 for empty dataset
        self.assertEqual(accuracy, 0.0)
        
    @patch('experiment_orchestrator.load_dataset')
    def test_single_example_dataset(self, mock_load):
        """Test handling of dataset with only one example"""
        mock_load.return_value = [self.mock_dataset[0]]
        
        with patch('experiment_orchestrator.create_few_shot_prompt') as mock_prompt:
            with patch('experiment_orchestrator.get_model_response'):
                with patch('experiment_orchestrator.parse_response'):
                    with patch('experiment_orchestrator.evaluate_answer', return_value=True):
                        accuracy = run_experiment('single_task', use_cot=True)
        
        # Should still work but with no test examples (only exemplars)
        self.assertEqual(accuracy, 0.0)  # No test examples to evaluate


if __name__ == '__main__':
    unittest.main()