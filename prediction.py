import cv2
import numpy as np
import json
import os
from datetime import datetime
import random
import hashlib

HISTORY_FILE = "history.json"

def analyze_video(video_path):
    """
    Analyze video for fertility prediction using OpenCV
    This is a demo implementation with simulated AI logic
    """
    try:
        # Create a deterministic seed based on video content for consistent results
        video_hash = get_video_hash(video_path)
        random.seed(video_hash)
        np.random.seed(video_hash % (2**32))  # Convert to 32-bit for numpy
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        fertility_scores = []
        frame_data = []
        feature_data = []  # Store multi-feature analysis data
        
        # Analyze frames (sample every 30 frames for efficiency)
        frame_idx = 0
        sample_interval = max(1, int(fps)) if fps > 0 else 30
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                # Perform multi-feature analysis on frame
                features = analyze_frame_multi_feature(frame, frame_idx / fps if fps > 0 else frame_idx, video_hash)
                fertility_score = features['fertility_score']
                fertility_scores.append(fertility_score)
                
                frame_data.append({
                    'timestamp': frame_idx / fps if fps > 0 else frame_idx,
                    'fertility_score': fertility_score
                })
                
                feature_data.append({
                    'timestamp': frame_idx / fps if fps > 0 else frame_idx,
                    'motion_score': features['motion_score'],
                    'posture_score': features['posture_score'],
                    'behavior_score': features['behavior_score'],
                    'physical_score': features['physical_score'],
                    'estrus_score': features['estrus_score'],
                    'fertility_score': fertility_score
                })
            
            frame_idx += 1
        
        cap.release()
        
        if not fertility_scores:
            return None
        
        # Calculate overall fertility percentage
        avg_fertility = np.mean(fertility_scores)
        
        # Calculate feature averages
        avg_motion = np.mean([f['motion_score'] for f in feature_data]) if feature_data else 0
        avg_posture = np.mean([f['posture_score'] for f in feature_data]) if feature_data else 0
        avg_behavior = np.mean([f['behavior_score'] for f in feature_data]) if feature_data else 0
        avg_physical = np.mean([f['physical_score'] for f in feature_data]) if feature_data else 0
        avg_estrus = np.mean([f['estrus_score'] for f in feature_data]) if feature_data else 0
        
        # Generate recommendation
        recommendation = generate_recommendation(avg_fertility)
        
        # Generate detailed feature analysis
        feature_analysis = {
            'motion': avg_motion,
            'posture': avg_posture,
            'behavior': avg_behavior,
            'physical_condition': avg_physical,
            'estrus_behavior': avg_estrus
        }
        
        return {
            'fertility_percentage': avg_fertility,
            'recommendation': recommendation,
            'frame_data': frame_data,
            'video_duration': duration,
            'frames_analyzed': len(fertility_scores),
            'feature_analysis': feature_analysis,
            'detailed_features': feature_data
        }
    
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return None

def get_video_hash(video_path):
    """Generate a hash of the video file for deterministic seeding"""
    hash_md5 = hashlib.md5()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    # Convert hex digest to integer for seeding
    return int(hash_md5.hexdigest(), 16) % (2**31)  # Convert to 32-bit int

def analyze_frame_multi_feature(frame, timestamp, video_hash):
    """
    Analyze individual frame for multiple fertility indicators
    Enhanced implementation with 5 key feature categories
    """
    # Set seed based on video hash and frame timestamp for consistency
    frame_seed = (video_hash + int(timestamp * 1000)) % (2**31)
    random.seed(frame_seed)
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Motion analysis - Activity level indicators
    motion_score = simulate_motion_analysis(gray)
    
    # 2. Posture analysis - Physical positioning indicators
    posture_score = simulate_posture_analysis(gray)
    
    # 3. Behavioral indicators - Estrus behavior patterns
    behavior_score = simulate_behavior_analysis(gray, timestamp)
    
    # 4. Physical condition - Body condition scoring
    physical_score = simulate_physical_analysis(gray)
    
    # 5. Estrus behavior - Specific estrus indicators
    estrus_score = simulate_estrus_analysis(gray, timestamp)
    
    # Combine scores with weighted average for final fertility score
    fertility_score = (
        motion_score * 0.2 +      # 20% weight
        posture_score * 0.15 +    # 15% weight
        behavior_score * 0.25 +   # 25% weight (most important)
        physical_score * 0.2 +    # 20% weight
        estrus_score * 0.2        # 20% weight
    )
    
    # Add minimal randomness to make it more realistic but keep it consistent
    # Using the frame seed to ensure consistency
    np.random.seed(frame_seed)
    fertility_score += np.random.uniform(-1, 1)
    
    # Ensure score is within valid range
    fertility_score = max(0, min(100, fertility_score))
    
    return {
        'fertility_score': fertility_score,
        'motion_score': motion_score,
        'posture_score': posture_score,
        'behavior_score': behavior_score,
        'physical_score': physical_score,
        'estrus_score': estrus_score
    }

def simulate_motion_analysis(gray_frame):
    """Simulate motion-based fertility analysis"""
    # Calculate frame statistics as proxy for motion
    mean_intensity = np.mean(gray_frame)
    std_intensity = np.std(gray_frame)
    
    # Simulate fertility correlation with activity level
    activity_score = (std_intensity / 255.0) * 100
    
    # Higher activity might indicate estrus behavior
    base_score = 60 + (activity_score * 0.3)
    
    return base_score

def simulate_posture_analysis(gray_frame):
    """Simulate posture-based fertility analysis"""
    # Use edge detection as proxy for posture analysis
    edges = cv2.Canny(gray_frame, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Simulate correlation between posture and fertility
    posture_score = 50 + (edge_density * 500)
    
    return min(100, posture_score)

def simulate_behavior_analysis(gray_frame, timestamp):
    """Simulate behavioral analysis over time"""
    # Simulate time-based behavioral patterns
    time_factor = np.sin(timestamp * 0.1) * 10  # Cyclical behavior
    
    # Deterministic behavioral indicators based on timestamp
    restlessness = (np.sin(timestamp * 0.5) * 10) + 10  # More deterministic
    social_behavior = (np.cos(timestamp * 0.3) * 7.5) + 7.5  # More deterministic
    
    behavior_score = 65 + time_factor + restlessness + social_behavior
    
    return max(0, min(100, behavior_score))

def simulate_physical_analysis(gray_frame):
    """Simulate physical condition analysis"""
    # Analyze texture and contrast as proxy for body condition
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
    texture_score = np.var(laplacian) / 1000
    
    # Analyze brightness distribution
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    brightness_score = np.sum(hist[-85:]) / np.sum(hist) * 100  # Bright areas
    
    # Combine physical indicators
    physical_score = 55 + (texture_score * 0.5) + (brightness_score * 0.3)
    
    return max(0, min(100, physical_score))

def simulate_estrus_analysis(gray_frame, timestamp):
    """Simulate specific estrus behavior analysis"""
    # Estrus behaviors often show specific patterns
    # Analyze for cyclical patterns that indicate estrus
    
    # Frequency of movement patterns (deterministic)
    frequency_score = abs(np.sin(timestamp * 0.3)) * 30
    
    # Intensity variations that indicate estrus
    intensity_var = np.std(gray_frame) / 255.0 * 100
    
    # Estrus typically shows higher activity in specific patterns
    estrus_indicators = frequency_score + intensity_var
    
    estrus_score = 50 + estrus_indicators
    
    # Add minimal deterministic variation
    estrus_score += np.sin(timestamp * 0.7) * 5
    
    return max(0, min(100, estrus_score))

def generate_recommendation(fertility_percentage):
    """Generate recommendation based on fertility score"""
    if fertility_percentage >= 90:
        return "Inseminate immediately - Peak fertility detected. Optimal window for maximum conception rates."
    elif fertility_percentage >= 85:
        return "Inseminate within 12 hours - High fertility detected. Excellent conception probability."
    elif fertility_percentage >= 80:
        return "Inseminate within 24 hours - Strong fertility signs. Good conception prospects."
    elif fertility_percentage >= 75:
        return "Monitor closely and prepare for insemination - Moderate to high fertility."
    elif fertility_percentage >= 70:
        return "Continue monitoring - Approaching fertile period. Watch for behavioral changes."
    elif fertility_percentage >= 65:
        return "Monitor closely - Early fertility signs detected. Estrus behavior may develop."
    elif fertility_percentage >= 60:
        return "Observe regularly - Potential fertility indicators. Continue routine monitoring."
    elif fertility_percentage >= 55:
        return "Continue monitoring - Subtle signs present. Maintain regular observation schedule."
    elif fertility_percentage >= 50:
        return "Monitor periodically - Minimal fertility indicators. Routine checks sufficient."
    else:
        return "Not fertile - Recheck in 24-48 hours. No estrus behavior detected at this time."

def save_prediction(username, video_filename, fertility_percentage, recommendation):
    """Save prediction to history file"""
    history = load_history()
    
    prediction = {
        'username': username,
        'timestamp': datetime.now().isoformat(),
        'video_filename': video_filename,
        'fertility_percentage': fertility_percentage,
        'recommendation': recommendation
    }
    
    history.append(prediction)
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def load_history():
    """Load prediction history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def clear_history():
    """Clear all prediction history"""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def get_user_history(username):
    """Get prediction history for specific user"""
    history = load_history()
    return [h for h in history if h.get('username') == username]