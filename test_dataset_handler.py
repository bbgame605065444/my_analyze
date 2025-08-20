import unittest
from dataset_handler import load_dataset


class TestDatasetHandler(unittest.TestCase):
    
    def test_load_gsm8k_dataset(self):
        """Test loading GSM8K math dataset"""
        dataset = load_dataset('gsm8k')
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        # Check required fields
        sample = dataset[0]
        self.assertIn('question', sample)
        self.assertIn('answer', sample)
        self.assertIn('chain_of_thought', sample)
        
        # Verify it's a math problem
        self.assertIsInstance(sample['question'], str)
        self.assertIsInstance(sample['answer'], str)
        
    def test_load_csqa_dataset(self):
        """Test loading CommonsenseQA dataset"""
        dataset = load_dataset('csqa')
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        sample = dataset[0]
        self.assertIn('question', sample)
        self.assertIn('answer', sample)
        self.assertIn('chain_of_thought', sample)
        
    def test_load_last_letter_dataset(self):
        """Test loading Last Letter Concatenation dataset"""
        dataset = load_dataset('last_letter_concatenation')
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        sample = dataset[0]
        self.assertIn('question', sample)
        self.assertIn('answer', sample)
        self.assertIn('chain_of_thought', sample)
        
    def test_invalid_task_name(self):
        """Test handling of invalid task names"""
        with self.assertRaises(ValueError):
            load_dataset('invalid_task')
            
    def test_dataset_structure_consistency(self):
        """Test that all datasets have consistent structure"""
        tasks = ['gsm8k', 'csqa', 'last_letter_concatenation']
        
        for task in tasks:
            dataset = load_dataset(task)
            for item in dataset:
                self.assertIn('question', item)
                self.assertIn('answer', item)
                self.assertIn('chain_of_thought', item)
                self.assertIsInstance(item['question'], str)
                self.assertIsInstance(item['answer'], str)
                self.assertIsInstance(item['chain_of_thought'], str)


if __name__ == '__main__':
    unittest.main()