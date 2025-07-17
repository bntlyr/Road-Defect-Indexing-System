"""
Simple RDIS Accuracy Test Script
This script provides a quick way to test the accuracy of the Road Defect Indexing System.
"""

import os
import sys
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the comprehensive test script
try:
    from test_severity_calculator_accuracy import AccuracyEvaluator
    
    def quick_test():
        """
        Quick test function for RDIS accuracy evaluation.
        """
        print("RDIS Accuracy Test")
        print("=" * 30)
        
        # Default test parameters
        test_directory = r'C:\Users\rafab\OneDrive\Desktop\accuracy test'
        model_path = r'src\models\road_defect.pt'
        
        # Check if test directory exists
        if not os.path.exists(test_directory):
            print(f"ERROR: Test directory not found: {test_directory}")
            print("Please ensure the directory exists and contains test images with annotations.")
            return
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"WARNING: Model file not found: {model_path}")
            print("Will attempt to run without YOLO model (limited functionality)")
            model_path = None
        
        # Ask user for test mode
        print("\nChoose test mode:")
        print("1. Results only (fast, no visualizations)")
        print("2. Results + Enhanced visualizations (shows masking + bounding boxes)")
        print("3. Legacy mode (original visualizations)")
        
        choice = input("Enter your choice (1-3, default=2): ").strip()
        if not choice:
            choice = "2"
        
        # Initialize evaluator
        try:
            evaluator = AccuracyEvaluator(test_directory, model_path)
            print(f"Test directory: {test_directory}")
            print(f"Model path: {model_path}")
            print()
            
            if choice == "1":
                print("Running results-only evaluation...")
                results = evaluator.run_evaluation_with_results_only(confidence_threshold=0.30)
            elif choice == "3":
                print("Running legacy evaluation with original visualizations...")
                results = evaluator.run_full_evaluation(confidence_threshold=0.30, show_comparisons=True)
            else:  # choice == "2" or default
                print("Running evaluation with enhanced visualizations...")
                print("Note: Confidence thresholds are not displayed in visualizations")
                results = evaluator.run_full_evaluation(confidence_threshold=0.30, show_comparisons=True)
            
            if results:
                print("\nTEST RESULTS SUMMARY:")
                print("-" * 30)
                det_acc = results['detection_performance']['accuracy_percentage']
                mask_acc = results['masking_performance']['accuracy_percentage']
                f1_score = results['detection_performance']['f1_score']
                mean_iou = results['masking_performance']['mean_iou']
                
                print(f"Detection Accuracy: {det_acc:.1f}%")
                print(f"Masking Accuracy: {mask_acc:.1f}%")
                print(f"F1-Score: {f1_score:.3f}")
                print(f"Mean IoU: {mean_iou:.3f}")
                
                if choice != "1":
                    print()
                    print("Enhanced visualizations include:")
                    print("• Original image")
                    print("• Ground truth annotations")
                    print("• Detected bounding boxes (without confidence values)")
                    print("• Defect mask overlay")
                    print("• Combined bounding boxes + mask")
                    print("• Detailed accuracy metrics")
                
                print()
                print("Success: Evaluation completed successfully!")
                print("Success: Detailed results saved to JSON and TXT files")
                
                if choice != "1":
                    print("Success: Enhanced visualization images saved with timestamp")
            else:
                print("Error: Evaluation failed - no results obtained")
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            logging.exception("Full traceback:")
    
    if __name__ == "__main__":
        quick_test()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and the main test script is available.")