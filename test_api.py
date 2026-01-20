#!/usr/bin/env python3
"""
Example script to test the Image Classification API

Usage:
    python test_api.py path/to/image.jpg
"""

import sys
import time
import requests
from PIL import Image


API_BASE = "http://localhost:8004"


def create_test_image(filename="test_image.jpg"):
    """Create a simple test image if none provided"""
    print(f"Creating test image: {filename}")
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    img.save(filename)
    return filename


def upload_image(image_path):
    """Upload an image for classification"""
    print(f"\n Uploading {image_path}...")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_BASE}/predictions/", files=files)
    
    if response.status_code == 201:
        data = response.json()
        print(f"Success! Prediction ID: {data['id']}")
        return data['id']
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None


def get_prediction(prediction_id, wait=True):
    """Get prediction results, optionally waiting for completion"""
    print(f"\n Getting prediction {prediction_id}...")
    
    if wait:
        print("Waiting for classification to complete...")
        max_attempts = 30
        for i in range(max_attempts):
            response = requests.get(f"{API_BASE}/predictions/{prediction_id}/")
            data = response.json()
            
            if data.get('top_prediction'):
                break
            
            time.sleep(1)
            print(f"Attempt {i+1}/{max_attempts}...")
        else:
            print("⚠️Timeout waiting for results")
            return None
    else:
        response = requests.get(f"{API_BASE}/predictions/{prediction_id}/")
        data = response.json()
    
    return data


def print_results(data):
    """Pretty print the prediction results"""
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Filename: {data['filename']}")
    print(f"Created: {data['created_at']}")
    print()
    
    if data.get('top_prediction'):
        print(f"Top Prediction: {data['top_prediction']}")
        print(f"   Confidence: {data['confidence']:.2%}")
        print()
        
        if data.get('all_predictions'):
            print("Top 5 Predictions:")
            for i, pred in enumerate(data['all_predictions'][:5], 1):
                print(f"   {i}. {pred['label']:<25} {pred['confidence']:.2%}")
    else:
        print("Classification not yet complete")
    
    print("="*60)


def list_all_predictions():
    """List all predictions in the database"""
    print("\nAll Predictions:")
    response = requests.get(f"{API_BASE}/predictions/")
    predictions = response.json()
    
    if not predictions:
        print("   No predictions found")
        return
    
    for pred in predictions:
        status = 'OK' if pred.get('top_prediction') else "Pending"
        prediction = pred.get('top_prediction', 'Processing...')
        print(f"   {status} ID {pred['id']}: {pred['filename']} → {prediction}")


def delete_prediction(prediction_id):
    """Delete a prediction"""
    print(f"\nDeleting prediction {prediction_id}...")
    response = requests.delete(f"{API_BASE}/predictions/{prediction_id}/")
    
    if response.status_code == 200:
        print("Deleted successfully")
    else:
        print(f"Error: {response.status_code}")


def check_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE}/ping")
        if response.status_code == 200:
            print("API is running!")
            return True
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API. Is it running?")
        print(f"   Expected at: {API_BASE}")
        return False


def main():
    """Main test workflow"""
    print("="*60)
    print("IMAGE CLASSIFICATION API TESTER")
    print("="*60)
    
    # Check if API is running
    if not check_health():
        sys.exit(1)
    
    # Get or create image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nUsing provided image: {image_path}")
    else:
        image_path = create_test_image()
    
    # Upload image
    prediction_id = upload_image(image_path)
    if not prediction_id:
        sys.exit(1)
    
    # Get results
    results = get_prediction(prediction_id, wait=True)
    if results:
        print_results(results)
    
    # List all predictions
    list_all_predictions()
    
    # Optional: Clean up
    # delete_prediction(prediction_id)


def interactive_demo():
    """Interactive demo mode"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("  1. Upload new image")
        print("  2. Check prediction")
        print("  3. List all predictions")
        print("  4. Delete prediction")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            path = input("Enter image path (or press Enter for test image): ").strip()
            if not path:
                path = create_test_image()
            pred_id = upload_image(path)
            if pred_id:
                results = get_prediction(pred_id, wait=True)
                if results:
                    print_results(results)
        
        elif choice == "2":
            pred_id = input("Enter prediction ID: ").strip()
            if pred_id.isdigit():
                results = get_prediction(int(pred_id), wait=False)
                if results:
                    print_results(results)
        
        elif choice == "3":
            list_all_predictions()
        
        elif choice == "4":
            pred_id = input("Enter prediction ID to delete: ").strip()
            if pred_id.isdigit():
                delete_prediction(int(pred_id))
        
        elif choice == "5":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        main()