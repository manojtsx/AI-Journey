#!/usr/bin/env python3
"""
Main execution script for the spam detector pipeline
Runs all steps in the correct order: data preparation -> training -> testing -> app demo
"""

print("=== Spam Detector Pipeline ===\n")

print("Step 1: Data Preparation")
print("-" * 30)
import data_preparation
print("✓ Data prepared and vectorizer created\n")

print("Step 2: Model Training")
print("-" * 30)
import training_data
print("✓ Model training completed\n")

print("Step 3: Model Evaluation")
print("-" * 30)
import accuracy_test
print("✓ Model evaluation completed\n")

print("Step 4: Integration Test")
print("-" * 30)
import test_integration
print("✓ Integration test completed\n")

print("=== Pipeline Complete ===")
print("All modules are properly connected and working!")
print("\nTo run the interactive app, use: python app.py")
print("To run individual steps, use:")
print("  python data_preparation.py")
print("  python training_data.py") 
print("  python accuracy_test.py")
