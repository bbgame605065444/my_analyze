import unittest
from unittest.mock import patch, Mock
from io import StringIO
import sys


class TestMain(unittest.TestCase):
    
    @patch('builtins.print')
    @patch('main.run_experiment')
    def test_main_execution_flow(self, mock_run_experiment, mock_print):
        """Test the main execution flow"""
        # Mock experiment results
        mock_run_experiment.side_effect = [
            0.75,  # gsm8k standard
            0.90,  # gsm8k CoT
            0.60,  # csqa standard  
            0.80,  # csqa CoT
            0.45,  # last_letter standard
            0.70   # last_letter CoT
        ]
        
        # Import and run main
        import main
        
        # Verify all experiments were run
        expected_calls = [
            (('gsm8k', False),),
            (('gsm8k', True),),
            (('csqa', False),),
            (('csqa', True),),
            (('last_letter_concatenation', False),),
            (('last_letter_concatenation', True),)
        ]
        
        actual_calls = mock_run_experiment.call_args_list
        self.assertEqual(len(actual_calls), 6)
        
        for expected, actual in zip(expected_calls, actual_calls):
            self.assertEqual(expected, actual)
            
    @patch('builtins.print')
    @patch('main.run_experiment')
    def test_results_formatting(self, mock_run_experiment, mock_print):
        """Test that results are formatted correctly"""
        # Mock specific results
        mock_run_experiment.side_effect = [
            0.75,  # gsm8k standard
            0.90,  # gsm8k CoT
            0.60,  # csqa standard
            0.80,  # csqa CoT
            0.45,  # last_letter standard
            0.70   # last_letter CoT
        ]
        
        # Import main to trigger execution
        import main
        
        # Check that print was called with expected format
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        
        # Should contain results for each task
        result_lines = [call for call in print_calls if 'Standard:' in call or 'CoT:' in call]
        
        # Should have 6 result lines (2 per task)
        self.assertEqual(len(result_lines), 6)
        
        # Check specific formatting
        gsm8k_lines = [line for line in result_lines if 'gsm8k' in line]
        self.assertEqual(len(gsm8k_lines), 2)
        
    @patch('builtins.print') 
    @patch('main.run_experiment')
    def test_improvement_calculation(self, mock_run_experiment, mock_print):
        """Test that CoT improvements are calculated correctly"""
        # Mock results with clear improvement
        mock_run_experiment.side_effect = [
            0.50,  # gsm8k standard
            0.80,  # gsm8k CoT (60% improvement)
            0.40,  # csqa standard
            0.60,  # csqa CoT (50% improvement)
            0.30,  # last_letter standard
            0.45   # last_letter CoT (50% improvement)
        ]
        
        import main
        
        # Check that improvements are mentioned in output
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        full_output = '\n'.join(print_calls)
        
        # Should mention improvements
        self.assertIn('improvement', full_output.lower())
        
    @patch('main.run_experiment')
    def test_task_list_completeness(self, mock_run_experiment):
        """Test that all expected tasks are included"""
        mock_run_experiment.return_value = 0.5
        
        import main
        
        # Check that all 3 tasks are tested
        called_tasks = set()
        for call in mock_run_experiment.call_args_list:
            task_name = call[0][0]
            called_tasks.add(task_name)
            
        expected_tasks = {'gsm8k', 'csqa', 'last_letter_concatenation'}
        self.assertEqual(called_tasks, expected_tasks)
        
    @patch('builtins.print')
    @patch('main.run_experiment') 
    def test_error_handling(self, mock_run_experiment, mock_print):
        """Test error handling in main execution"""
        # First experiment fails, others succeed
        mock_run_experiment.side_effect = [
            Exception("API Error"),
            0.80,  # gsm8k CoT
            0.60,  # csqa standard
            0.80,  # csqa CoT
            0.45,  # last_letter standard
            0.70   # last_letter CoT
        ]
        
        # Should handle the error gracefully
        try:
            import main
        except Exception:
            self.fail("Main should handle experiment errors gracefully")
            
    @patch('builtins.print')
    @patch('main.run_experiment')
    def test_summary_statistics(self, mock_run_experiment, mock_print):
        """Test that summary statistics are calculated correctly"""
        mock_run_experiment.side_effect = [
            0.70,  # gsm8k standard
            0.85,  # gsm8k CoT
            0.60,  # csqa standard
            0.75,  # csqa CoT
            0.40,  # last_letter standard
            0.55   # last_letter CoT
        ]
        
        import main
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        full_output = '\n'.join(print_calls)
        
        # Should include average performance
        self.assertIn('average', full_output.lower())
        
    def test_module_imports(self):
        """Test that all required modules can be imported"""
        try:
            from dataset_handler import load_dataset
            from prompt_engineer import create_few_shot_prompt
            from model_interface import get_model_response
            from response_parser import parse_response, evaluate_answer
            from experiment_orchestrator import run_experiment
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")


if __name__ == '__main__':
    unittest.main()