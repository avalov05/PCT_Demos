import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from hand_tracker_module import SimpleHandTracker

class SimpleHandDemo:
    def __init__(self):
        # Use the tracker module
        self.tracker = SimpleHandTracker(camera_index=0, max_hands=1)
        
        # Display settings
        self.screen_width = 1920
        self.screen_height = 1080
        cv2.namedWindow('Hand Ball', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Hand Ball', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Ball state - start in center
        self.last_ball_x = self.screen_width // 2
        self.last_ball_y = self.screen_height // 2
        
        # Sine wave parameters
        self.sine_amplitude = 500
        self.sine_frequency = 0.1
        self.start_time = time.time()
        
        # Tracking data for graph
        self.tracked_x = []
        self.tracked_y = []
        self.tracked_times = []

    def calculate_sine_disturbance(self, elapsed_time):
        return int(self.sine_amplitude * np.sin(2 * np.pi * self.sine_frequency * elapsed_time))

    def update_ball_position(self, hand_coords):
        elapsed_time = time.time() - self.start_time
        sine_offset = self.calculate_sine_disturbance(elapsed_time)

        if hand_coords:
            # Get first hand coordinates
            center_x, center_y = hand_coords[0]
            
            # Calculate relative position from center
            relative_x = center_x - 160
            relative_y = center_y - 120
            
            # Reverse X direction and scale to screen
            reversed_relative_x = -relative_x 
            ball_x = self.screen_width // 2 + int(reversed_relative_x * self.screen_width / 320) + sine_offset
            ball_y = self.screen_height // 2
            
            self.last_ball_x = ball_x
            self.last_ball_y = ball_y
            
            # Store tracking data for graph
            self.tracked_x.append(reversed_relative_x * self.screen_width / 320)
            self.tracked_y.append(relative_y)
            self.tracked_times.append(elapsed_time)
            
            return ball_x, ball_y
        else:
            # No hand detected, use last position with sine wave
            return self.last_ball_x + sine_offset, self.last_ball_y

    def draw_scene(self, ball_x, ball_y):
        # Create white background
        ball_frame = np.ones((self.screen_height, self.screen_width, 3), dtype=np.uint8) * 255
        
        # Draw green target ball at the center
        target_radius = int(60 * self.screen_width / 1920)
        cv2.circle(ball_frame, (self.screen_width // 2, self.screen_height // 2), target_radius, (0, 200, 0), -1)
        
        # Draw black tracked ball
        ball_radius = int(30 * self.screen_width / 1920)
        cv2.circle(ball_frame, (ball_x, ball_y), ball_radius, (0, 0, 0), -1)
        
        cv2.imshow('Hand Ball', ball_frame)

    def show_final_graph(self):
        if len(self.tracked_x) > 1:
            tracked_times_array = np.array(self.tracked_times)
            sine_data = self.sine_amplitude * np.sin(2 * np.pi * self.sine_frequency * tracked_times_array)
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.tracked_times, self.tracked_x, 'r-', linewidth=2, label='Hand Position')
            plt.plot(self.tracked_times, sine_data, 'b-', linewidth=2, label='Sine Disturbance')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Center')
            plt.xlabel('Time (seconds)')
            plt.ylabel('X Position (Relative to Center)')
            plt.title('Hand Position and Sine Disturbance Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def run(self):
        print("Simple Hand Demo - Press 'q' to quit")
        
        while True:
            # Get hand coordinates using the tracker module
            hand_coords = self.tracker.get_hand_coordinates()
            
            # Update ball position
            ball_x, ball_y = self.update_ball_position(hand_coords)
            
            # Draw the scene
            self.draw_scene(ball_x, ball_y)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()

    def cleanup(self):
        self.tracker.cleanup()
        cv2.destroyAllWindows()
        self.show_final_graph()
        print("Demo stopped")

if __name__ == "__main__":
    demo = SimpleHandDemo()
    demo.run()
