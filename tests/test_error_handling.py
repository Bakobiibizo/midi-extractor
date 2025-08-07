#!/usr/bin/env python3
"""
Test error handling scenarios for the MIDI Generator application.

This script tests various error conditions to ensure they are properly handled
and reported to the user.
"""
import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
src_dir = str(Path(__file__).parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Local imports
from utils.exceptions import (
    ConfigurationError,
    DatasetError,
    ModelError,
    MIDIGeneratorError
)
from cli import (
    command_index,
    command_filter,
    command_visualize,
    command_train,
    parse_args
)

class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.test_dir) / "test_data"
        self.test_data_dir.mkdir()
        
        # Create a minimal valid dataset
        self.valid_json = self.test_data_dir / "valid_dataset.json"
        self.valid_json.write_text('{"tracks": {"track1": {"path": "test.wav"}}}')
        
        # Create an invalid JSON file
        self.invalid_json = self.test_data_dir / "invalid.json"
        self.invalid_json.write_text('{invalid json')
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_nonexistent_directory(self):
        """Test handling of non-existent directory in command_index."""
        # Set up test data
        test_dir = "/nonexistent/path"
        args = MagicMock()
        args.dataset = test_dir
        args.output = None
        args.debug = False
        
        # Test that the correct exception is raised with the expected message
        with self.assertRaises(DatasetError) as context:
            command_index(args)
        self.assertIn("Dataset directory not found", str(context.exception))
    
    def test_invalid_json(self):
        """Test handling of invalid JSON in command_filter."""
        # Set up test data
        test_file = str(self.invalid_json)
        output_file = str(self.test_data_dir / "output.json")
        args = MagicMock()
        args.index_file = test_file
        args.output = output_file
        args.threshold = -35.0
        args.debug = False
        
        # Ensure the test file exists with invalid JSON
        test_path = Path(test_file)
        if not test_path.exists():
            test_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.write_text("{invalid json}")
        
        # Test that the correct exception is raised with the expected message
        with self.assertRaises(DatasetError) as context:
            command_filter(args)
        self.assertIn("Failed to parse JSON file", str(context.exception))
    
    def test_visualize_nonexistent_file(self):
        """Test handling of non-existent file in command_visualize."""
        # Set up test data
        test_file = "/nonexistent/file.json"
        output_dir = str(Path(self.test_dir) / "output")
        args = MagicMock()
        args.index_file = test_file
        args.output_dir = output_dir
        args.num_samples = 1
        args.max_len = 1000
        args.debug = False
        
        # Test that the correct exception is raised with the expected message
        with self.assertRaises(DatasetError) as context:
            command_visualize(args)
        self.assertIn("Index file not found", str(context.exception))
    
    def test_invalid_arguments(self):
        """Test handling of invalid command line arguments."""
        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['script.py', 'invalid-command']):
                parse_args()
    
    def test_missing_required_args(self):
        """Test handling of missing required arguments."""
        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['script.py', 'index']):  # Missing dataset argument
                parse_args()

if __name__ == "__main__":
    unittest.main()
