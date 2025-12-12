import unittest
from compiler import compile_pml

class TestPMLCompiler(unittest.TestCase):
    def test_compile_simple_module(self):
        pml = "module test { // test module }"
        result = compile_pml(pml)
        self.assertIn("Generated from PML", result)
        self.assertIn("module test", result)

    def test_compile_contract(self):
        pml = "contract test_contract { interface { } invariants { } }"
        result = compile_pml(pml)
        self.assertIn("Generated from PML", result)
        self.assertIn("contract test_contract", result)

    def test_compile_pipeline(self):
        pml = "pipeline test_pipeline { step1: module_a.do_something() }"
        result = compile_pml(pml)
        self.assertIn("Generated from PML", result)
        self.assertIn("pipeline test_pipeline", result)

if __name__ == "__main__":
    unittest.main()
