"""
Quick results-only test for RDIS accuracy evaluation.
This script runs the evaluation and shows only numerical results without any visualizations.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def results_only_test():
    """
    Run evaluation and show only results.
    """
    print("RDIS Accuracy Test - Results Only")
    print("=" * 40)
    
    # Default test parameters
    test_directory = r'C:\Users\rafab\OneDrive\Desktop\accuracy test'
    model_path = r'src\models\road_defect.pt'
    
    # Check if test directory exists
    if not os.path.exists(test_directory):
        print(f"ERROR: Test directory not found: {test_directory}")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found: {model_path}")
        print("Will attempt to run without YOLO model (limited functionality)")
        model_path = None
    
    print(f"Test directory: {test_directory}")
    print(f"Model path: {model_path}")
    print("\nImporting evaluation modules...")
    
    # Import the comprehensive test script
    try:
        from test_severity_calculator_accuracy import AccuracyEvaluator
        print("Successfully imported AccuracyEvaluator")
        
        # Initialize evaluator and run results-only evaluation
        print("Initializing evaluator...")
        evaluator = AccuracyEvaluator(test_directory, model_path)
        print("Evaluator initialized successfully")
        
        print("\nRunning evaluation (results only)...")
        results = evaluator.run_evaluation_with_results_only(confidence_threshold=0.30)
        
        if results:
            print("\nRESULTS SUMMARY:")
            print("-" * 30)
            det_acc = results['detection_performance']['accuracy_percentage']
            mask_acc = results['masking_performance']['accuracy_percentage']
            f1_score = results['detection_performance']['f1_score']
            mean_iou = results['masking_performance']['mean_iou']
            
            print(f"Detection Accuracy: {det_acc:.1f}%")
            print(f"Masking Accuracy: {mask_acc:.1f}%")
            print(f"F1-Score: {f1_score:.3f}")
            print(f"Mean IoU: {mean_iou:.3f}")
            print(f"Total Images: {results['test_summary']['total_images']}")
            
            print("\nSuccess: Evaluation completed successfully!")
            print("Success: Detailed results saved to JSON and TXT files")
        else:
            print("Error: Evaluation failed - no results obtained")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed and the main test script is available.")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    results_only_test()
