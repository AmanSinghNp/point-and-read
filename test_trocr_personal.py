import os
import csv
from predictor import predict_with_confidence

DATA_DIR = r"d:\Deep Learning Project\TextRecognition\point-and-read\data\personal_dataset\images"
LABELS_FILE = r"d:\Deep Learning Project\TextRecognition\point-and-read\data\personal_dataset\labels.csv"

def main():
    print("Testing TrOCR on Personal Dataset...")
    
    # Load labels
    labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["filename"]] = row["text"]
                
    # Loop over images
    total_images = 0
    correct_predictions = 0
    total_confidence = 0.0
    
    # We will use the default model ('base') config as per the app
    for filename in os.listdir(DATA_DIR):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        filepath = os.path.join(DATA_DIR, filename)
        
        try:
            # Running inference with TrOCR
            text, conf = predict_with_confidence(filepath)
            
            ground_truth = labels.get(filename, "Unknown")
            print(f"\n--- {filename} ---")
            print(f"Ground Truth : {ground_truth}")
            print(f"Prediction   : {text}")
            print(f"Confidence   : {conf:.4f}")
            
            total_images += 1
            total_confidence += conf
            if text.strip() == ground_truth.strip():
                correct_predictions += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if total_images > 0:
        print(f"\n======================================")
        print(f"Total Images   : {total_images}")
        print(f"Exact Matches  : {correct_predictions} / {total_images} ({(correct_predictions/total_images)*100:.2f}%)")
        print(f"Avg Confidence : {total_confidence/total_images:.4f}")
        print(f"======================================")

if __name__ == "__main__":
    main()
