import unittest
from zoning.term_extraction.eval_results import clean_string_units

class TestEval(unittest.TestCase):

    def test_clean_string_units(self):
        """
        testing the clean_string_units method used in eval_results.py. 
        This method extracts numerical values from a string, and converts them to square feet. 
        """
        self.assertEqual(clean_string_units("40,000 sq ft"),[40000.0])
        self.assertEqual(clean_string_units("2 acres"), [87120.0])
        self.assertEqual(clean_string_units("4 acres (subject to conditions); 5 acres (for PS-A district)"), [174240.0, 217800.0])
        self.assertEqual(clean_string_units("2 acres (60000 sq ft)"), [87120.0, 60000.0])
        self.assertEqual(clean_string_units("2 acres; 60000 sq ft"), [87120.0, 60000.0])
        self.assertEqual(clean_string_units("1 acre or 20 sq ft"), [43560.0, 20])
        self.assertEqual(clean_string_units("1.5 acre and 20 sq ft higher than x"), [65340.0, 20])
        self.assertEqual(clean_string_units("142 H acres"), [6185520.0])
        

if __name__ == '__main__':
    unittest.main()