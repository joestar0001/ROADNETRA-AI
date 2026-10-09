"""
ROADNETRA AI - Multi-Class Road Infrastructure & Defect Model Training
Problem Statement 03.1: Vision in the Wild (IEEE Hackathon 2026)
Team ESPADA - Bhawesh Kumar Gautam
"""

import os
import torch
from ultralytics import YOLO

def main():
    # 1. Device configuration
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon Metal Performance Shaders (MPS) GPU acceleration!")
    elif torch.cuda.is_available():
        device = 0
        print(f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU device.")

    # 2. Paths configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = os.path.join(BASE_DIR, "datasets/unified_infrastructure/data.yaml")
    MODEL_WEIGHTS = os.path.join(BASE_DIR, "models/yolov8s.pt")
    RUNS_DIR = os.path.join(BASE_DIR, "runs")
    
    print(f"\n==========================================")
    print(f"ROADNETRA AI - TRAINING PIPELINE READY")
    print(f"Dataset YAML: {DATA_YAML}")
    print(f"Base Weights: {MODEL_WEIGHTS}")
    print(f"Output Directory: {RUNS_DIR}")
    print(f"==========================================\n")

    # 3. Load pre-trained base model
    model = YOLO(MODEL_WEIGHTS)

    # 4. Execute Fine-Tuning with 'Vision in the Wild' Augmentation Parameters
    results = model.train(
        data=DATA_YAML,
        epochs=40,
        imgsz=640,
        batch=16,
        device=device,
        patience=10,             # Early stopping if validation loss stops improving
        
        # --- "Vision in the Wild" Field Robustness Hyperparameters ---
        hsv_h=0.015,             # Subtle hue adjustments
        hsv_s=0.7,               # Saturation jitter (simulates rain, wet road reflections & smog)
        hsv_v=0.4,               # Brightness jitter (simulates night CCTV, tunnels, bright sun glare)
        degrees=10.0,            # Camera pole sway & vehicle vibration
        translate=0.1,           # Bounding frame movement
        scale=0.5,               # Object scale variations (close vs distant defects)
        perspective=0.0005,      # Highway perspective distortion
        mosaic=1.0,              # Occlusion handling & multi-hazard recognition
        mixup=0.15,              # Blends background noise into scene
        # -------------------------------------------------------------
        
        project=RUNS_DIR,
        name="roadnetra_infra_v1",
        exist_ok=True,
        plots=True
    )

    print("\nTraining completed successfully!")
    
    # 5. Export best weights to models/pothole_best.pt
    best_weights = os.path.join(RUNS_DIR, "roadnetra_infra_v1", "weights", "best.pt")
    target_weights = os.path.join(BASE_DIR, "models", "pothole_best.pt")
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"Saved deployment checkpoint to: {target_weights}")

if __name__ == "__main__":
    main()
