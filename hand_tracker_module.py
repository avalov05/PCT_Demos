import cv2
import mediapipe as mp
from typing import List, Tuple, Optional

class SimpleHandTracker:
    #returns a list of (x, y) coordinates for each hand detected
    
    def __init__(self, camera_index: int = 0, max_hands: int = 2):
        """
        Args:
            camera_index: Which camera to use (0, 1, 2, etc.)
            max_hands: Maximum number of hands to track
        """
        self.camera_index = camera_index
        self.max_hands = max_hands
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def get_hand_coordinates(self) -> List[Tuple[int, int]]:
        ret, frame = self.cap.read()
        if not ret:
            return []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        coordinates = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
                
                center_x = int(sum(x_coords) / len(x_coords))
                center_y = int(sum(y_coords) / len(y_coords))
                
                coordinates.append((center_x, center_y))
        
        return coordinates
    
    def get_hand_coordinates_with_frame(self, frame: Optional[cv2.Mat] = None) -> Tuple[List[Tuple[int, int]], cv2.Mat]:
        """
        Get hand coordinates and return the frame as well.
        
        Args:
            frame: Optional frame to process (if None, captures from camera)
            
        Returns:
            Tuple of (coordinates_list, frame)
        """
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                return [], frame
        
        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        coordinates = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get all landmark coordinates
                h, w, _ = frame.shape
                x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
                
                # Calculate center of hand
                center_x = int(sum(x_coords) / len(x_coords))
                center_y = int(sum(y_coords) / len(y_coords))
                
                coordinates.append((center_x, center_y))
        
        return coordinates, frame
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        self.hands.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

# Convenience function for quick usage
def get_hand_coordinates(camera_index: int = 0, max_hands: int = 2) -> List[Tuple[int, int]]:
    """
    Quick function to get hand coordinates.
    
    Args:
        camera_index: Camera device index
        max_hands: Maximum number of hands to track
        
    Returns:
        List of (x, y) coordinates for detected hands
    """
    with SimpleHandTracker(camera_index, max_hands) as tracker:
        return tracker.get_hand_coordinates()

# Example usage
if __name__ == "__main__":
    print("Simple Hand Tracker Test")
    print("=" * 30)
    
    # Test single camera
    print("Testing camera 0...")
    while True:
        print( get_hand_coordinates(camera_index=0, max_hands=2))