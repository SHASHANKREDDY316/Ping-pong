import cv2
import numpy as np
import math
import time
import os

class HandTracker:
    """
    A class to detect and track the user's hand using basic computer vision
    techniques (skin color segmentation and face exclusion).
    """
    def __init__(self):
        # Load the pre-trained Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        """Detects faces in the frame to exclude them from skin detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Adjust scaleFactor and minNeighbors for better performance/accuracy
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        return faces

    def detect_hand_skin(self, frame, face_regions=None):
        """
        Detects skin-colored regions and masks out the face areas.
        Uses HSV color space for more robust skin detection.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Define HSV range for common skin tones (can be tuned)
        lower_skin = np.array([0, 15, 40], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Exclude face regions from the mask to prevent false positives
        if face_regions is not None:
            for (x, y, w, h) in face_regions:
                # Expand the exclusion margin slightly
                margin = 30
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                # Set the face area in the mask to black (0)
                mask[y1:y2, x1:x2] = 0

        # Apply morphological operations for noise reduction and smoothing
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return mask

    def find_hand_details(self, frame, face_regions):
        """Finds the largest skin contour, its center, bounding box, and potential fingertip."""
        mask = self.detect_hand_skin(frame, face_regions)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, None, None, None

        # Find the largest contour (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Filter out small contours (noise)
        if cv2.contourArea(max_contour) < 2000:
            return None, None, None, None

        # Convex Hull and Defects for fingertip detection
        hull = cv2.convexHull(max_contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            return None, None, None, None
            
        defects = cv2.convexityDefects(max_contour, hull)
        fingertips = []

        if defects is not None:
            # Look for deep, narrow valleys (defects) that might indicate space between fingers
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])
                
                # Check angle and depth of the defect
                angle = self._angle(far, start, end)
                if angle < 90 and d > 10000: # d > 10000 is a depth threshold
                    # The start/end points of the defect often lie near fingertips
                    fingertips.append(start)
                    fingertips.append(end)
            
            # Simple heuristic: filter points and take the highest one as the index finger
            fingertips = list(set(fingertips))
            if fingertips:
                # Sort by Y-coordinate and take the highest point (smallest Y value)
                fingertips = sorted(fingertips, key=lambda pt: pt[1])
                if fingertips:
                     # Take the single topmost point as the control point
                    fingertips = [fingertips[0]] 

        # Fallback: If no clear defect-based fingertips, use the topmost hull point
        if not fingertips:
            hull_points = cv2.convexHull(max_contour, returnPoints=True)
            topmost = tuple(hull_points[hull_points[:, :, 1].argmin()][0])
            fingertips = [topmost]

        # Calculate hand center (centroid)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        else:
            center = None

        # Bounding box of the hand
        x, y, w, h = cv2.boundingRect(max_contour)
        bbox = (x, y, w, h)
        
        # Return center, bounding box, contour, and the single best fingertip position
        return center, bbox, max_contour, fingertips

    def _angle(self, pt1, pt2, pt3):
        """Calculates the angle (in degrees) between three points."""
        a = np.array(pt1)
        b = np.array(pt2)
        c = np.array(pt3)
        ab = a - b
        cb = c - b
        
        # Calculate cosine using dot product
        # Added a small epsilon (1e-8) to avoid division by zero
        cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-8)
        
        # Clamp cosine to [-1, 1] for arccos stability
        angle = np.arccos(np.clip(cosine, -1, 1))
        return np.degrees(angle)

class Particle:
    """Represents a small visual particle for effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(2, 6)
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed
        self.life = 1.0  # Particle lifetime, from 1.0 to 0.0
        self.color = color
        self.size = np.random.randint(3, 8)
        
    def update(self):
        """Updates particle position and life."""
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.3 # Simple gravity effect
        self.life -= 0.02
        return self.life > 0

class AnimatedBallGame:
    """The core game logic and drawing routines."""
    def __init__(self, width, height, high_score=0):
        self.width = width
        self.height = height
        # Ball properties
        self.ball_x = width // 2
        self.ball_y = height // 2
        self.ball_vx = 0
        self.ball_vy = 0
        self.ball_radius = 25
        self.ball_color = (255, 100, 255) # Initial color
        self.ball_trail = []
        self.max_trail = 15
        
        # Game state and scoring
        self.targets = []
        self.score = 0
        self.high_score = high_score
        self.particles = []
        self.spawn_timer = 0
        self.game_started = False
        self.game_over = False
        self.game_duration = 60 # 60 seconds
        self.start_time = None
        self.remaining_time = self.game_duration
        
        # Animation effects
        self.glow_pulse = 0
        self.rainbow_offset = 0
        
        self.spawn_target() # Initial target
        
    def spawn_target(self):
        """Adds a new target to the game at a random position."""
        if len(self.targets) < 3: # Keep a maximum of 3 targets
            x = np.random.randint(100, self.width - 100)
            y = np.random.randint(100, self.height - 100)
            size = np.random.randint(30, 50)
            color = self.get_rainbow_color(np.random.randint(0, 360))
            self.targets.append({'x': x, 'y': y, 'size': size, 'color': color, 'pulse': 0})
            
    def get_rainbow_color(self, hue):
        """Converts a hue value (0-360) to an BGR color tuple."""
        hue = int(hue * 179 / 360) % 180 # OpenCV uses H range 0-179
        c = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)
        # Note: OpenCV BGR is returned, so the tuple is (B, G, R)
        return tuple(map(int, rgb[0, 0]))

    def start_timer(self):
        """Initializes the game timer."""
        self.start_time = time.time()
        
    def update_timer(self):
        """Updates the countdown timer and checks for game over."""
        if self.game_started and not self.game_over and self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.remaining_time = max(0, int(self.game_duration - elapsed))
            
            if self.remaining_time <= 0:
                self.game_over = True
                self.game_started = False
                if self.score > self.high_score:
                    self.high_score = self.score
                    # Save the new high score
                    save_high_score(self.high_score)

    def update_ball_from_finger(self, finger_pos):
        """
        Calculates ball velocity based on the distance and direction
        from the ball's center to the detected finger position.
        """
        if finger_pos is None:
            # Slow down if the hand is not visible
            self.ball_vx *= 0.95 
            self.ball_vy *= 0.95
            if abs(self.ball_vx) < 0.1: self.ball_vx = 0
            if abs(self.ball_vy) < 0.1: self.ball_vy = 0
            return
            
        fx, fy = finger_pos
        dx = fx - self.ball_x
        dy = fy - self.ball_y
        dist = math.sqrt(dx**2 + dy**2)
        
        # Apply force/speed towards the finger
        if dist > 10:
            speed = min(dist / 10, 15) # Max speed is 15
            self.ball_vx = (dx / dist) * speed
            self.ball_vy = (dy / dist) * speed
        else:
            # Stop if the finger is very close
            self.ball_vx = 0
            self.ball_vy = 0

    def update(self, finger_pos):
        """Updates game state: ball movement, collisions, targets, and particles."""
        if not self.game_started or self.game_over:
            return

        self.update_timer()
        self.update_ball_from_finger(finger_pos)

        # Update ball position
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Wall collisions
        if self.ball_x - self.ball_radius < 0:
            self.ball_x = self.ball_radius
            self.ball_vx *= -0.8
            self.create_particles(self.ball_x, self.ball_y, (100, 200, 255))
        elif self.ball_x + self.ball_radius > self.width:
            self.ball_x = self.width - self.ball_radius
            self.ball_vx *= -0.8
            self.create_particles(self.ball_x, self.ball_y, (100, 200, 255))

        if self.ball_y - self.ball_radius < 0:
            self.ball_y = self.ball_radius
            self.ball_vy *= -0.8
            self.create_particles(self.ball_x, self.ball_y, (100, 200, 255))
        elif self.ball_y + self.ball_radius > self.height:
            self.ball_y = self.height - self.ball_radius
            self.ball_vy *= -0.8
            self.create_particles(self.ball_x, self.ball_y, (100, 200, 255))

        # Trail update
        self.ball_trail.append((int(self.ball_x), int(self.ball_y)))
        if len(self.ball_trail) > self.max_trail:
            self.ball_trail.pop(0)

        # Target collisions
        for target in self.targets[:]:
            dist = math.sqrt((self.ball_x - target['x'])**2 + (self.ball_y - target['y'])**2)
            if dist < self.ball_radius + target['size']:
                self.score += 10
                self.targets.remove(target)
                self.create_particles(target['x'], target['y'], target['color'], 30) # Explosion effect
                self.spawn_target() # Spawn a new target immediately

        # Target spawning logic
        self.spawn_timer += 1
        if self.spawn_timer > 120: # Spawn a new target every 120 frames (~4 seconds)
            self.spawn_target()
            self.spawn_timer = 0

        # Update target pulsing animation
        for target in self.targets:
            target['pulse'] += 0.1

        # Particle system update
        self.particles = [p for p in self.particles if p.update()]
        
        # General visual effects update
        self.glow_pulse += 0.1
        self.rainbow_offset += 2

    def create_particles(self, x, y, color, count=15):
        """Generates particles at a given location."""
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def draw(self, frame):
        """Draws all game elements onto the given frame (which is now the static background)."""
        
        # 1. Draw animated background grid/dots
        grid_color = (10, 10, 40)
        for i in range(0, self.width, 50):
            cv2.line(frame, (i, 0), (i, self.height), grid_color, 1)
        for i in range(0, self.height, 50):
            cv2.line(frame, (0, i), (self.width, i), grid_color, 1)

        # 2. Draw particles
        for particle in self.particles:
            if particle.life > 0:
                size = int(particle.size * particle.life)
                # Create a fading color effect
                color = tuple(int(c * particle.life) for c in particle.color)
                cv2.circle(frame, (int(particle.x), int(particle.y)), size, color, -1)

        # 3. Draw targets
        for target in self.targets:
            pulse = math.sin(target['pulse']) * 5 + target['size'] # Pulsing size
            # Outer glow ring
            cv2.circle(frame, (target['x'], target['y']), int(pulse + 10), target['color'], 2)
            # Solid target center
            cv2.circle(frame, (target['x'], target['y']), int(pulse), target['color'], -1)
            # Inner highlight
            cv2.circle(frame, (target['x'], target['y']), int(pulse // 3), (255, 255, 255), -1)

        # 4. Draw ball trail
        for i, pos in enumerate(self.ball_trail):
            alpha = i / len(self.ball_trail)
            size = int(self.ball_radius * alpha * 0.8)
            # Fading trail color
            color = tuple(int(c * alpha * 0.8 + 50) for c in self.ball_color)
            cv2.circle(frame, pos, size, color, -1)

        # 5. Draw the ball
        # Outer glow effect
        glow_size = int(self.ball_radius + abs(math.sin(self.glow_pulse * 1.5)) * 8)
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), glow_size, (150, 50, 200), 4)

        # Rainbow color for the ball
        hue = (self.rainbow_offset % 360)
        ball_color = self.get_rainbow_color(hue)
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), self.ball_radius, ball_color, -1)
        
        # Ball highlight/shine
        cv2.circle(frame, (int(self.ball_x - 8), int(self.ball_y - 8)), 8, (255, 255, 255), -1)

        # 6. Draw the UI overlay (Score, Timer, Messages)
        self.draw_modern_ui(frame)
        
        return frame
        
    def draw_modern_ui(self, frame):
        """Draws the score, timer, high score, and game messages."""
        
        # Top panel overlay for better contrast
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, panel_height), (20, 20, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # SCORE display (Shadowed)
        score_text = f"SCORE: {self.score}"
        cv2.putText(frame, score_text, (18, 52), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 4)
        cv2.putText(frame, score_text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 150, 255), 2)
        
        # HIGH SCORE display
        high_score_text = f"HIGH SCORE: {self.high_score}"
        cv2.putText(frame, high_score_text, (self.width - 350, 52), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 215, 0), 2)
        
        # TIMER display
        timer_text = f"Time Left: {self.remaining_time:02d}s"
        cv2.putText(frame, timer_text, (self.width//2 - 80, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 200, 50), 3)

        # Start message
        if not self.game_started and not self.game_over:
            msg = "SHOW YOUR HAND TO START"
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_BOLD, 1.2, 3)[0]
            x = (self.width - text_size[0]) // 2
            y = self.height // 2
            
            # Animated background box for the message
            pulse = int(abs(math.sin(self.glow_pulse)) * 30) + 20
            cv2.rectangle(frame, (x - 20, y - 50), (x + text_size[0] + 20, y + 20), (pulse, pulse, pulse + 50), -1)
            
            # Message text (shadowed)
            cv2.putText(frame, msg, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
            cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
            
            # Sub-message
            sub_msg = "Use your index finger to control the ball"
            text_size2 = cv2.getTextSize(sub_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            x2 = (self.width - text_size2[0]) // 2
            cv2.putText(frame, sub_msg, (x2, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

        # Game Over message
        if self.game_over:
            box_width = 430
            box_height = 210
            x = (self.width - box_width) // 2
            y = (self.height - box_height) // 2
            
            # Game Over panel
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (10, 10, 10), -1)
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (255, 0, 120), 5) # Pink border
            
            # "GAME OVER!"
            msg = "GAME OVER!"
            msg_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 2.2, 6)[0]
            msg_x = x + (box_width - msg_size[0]) // 2
            msg_y = y + 65
            cv2.putText(frame, msg, (msg_x, msg_y), cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 0, 100), 6, cv2.LINE_AA)
            
            # Final Score
            score_text = f"Score: {self.score}"
            score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0]
            score_x = x + (box_width - score_size[0]) // 2
            score_y = msg_y + 55
            cv2.putText(frame, score_text, (score_x, score_y), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 0), 3, cv2.LINE_AA)
            
            # High Score
            high_score_text = f"High Score: {self.high_score}"
            hs_size = cv2.getTextSize(high_score_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
            hs_x = x + (box_width - hs_size[0]) // 2
            hs_y = score_y + 50
            cv2.putText(frame, high_score_text, (hs_x, hs_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 200, 50), 3, cv2.LINE_AA)
            
            # Restart prompt
            restart_text = "Time's up! Press R to restart"
            restart_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2)[0]
            restart_x = x + (box_width - restart_size[0]) // 2
            restart_y = hs_y + 45
            cv2.putText(frame, restart_text, (restart_x, restart_y), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 100, 255), 2, cv2.LINE_AA)
            
        # Controls footer
        controls = "Q: Quit | R: Reset | SPACE: Pause"
        cv2.putText(frame, controls, (10, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

def save_high_score(high_score, filename="high_score.txt"):
    """Saves the high score to a local file."""
    try:
        # NOTE: This saving mechanism works in a standalone Python environment
        # but may not be available in restricted execution environments.
        with open(filename, "w") as f:
            f.write(str(high_score))
    except Exception:
        # Fail silently if file writing is restricted
        pass

def load_high_score(filename="high_score.txt"):
    """Loads the high score from a local file."""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return int(f.read())
        except Exception:
            return 0
    else:
        return 0

def main():
    """Main function to run the camera, tracking, and game loop."""
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Initial read to determine frame size and check camera availability
    ret, initial_frame = cap.read()
    if not ret:
        print("Error: Cannot access camera or frame is empty.")
        # Attempt to create a mock frame size if camera fails, for testing only
        h, w = 720, 1280
        # return # Uncomment to strictly require camera
    else:
        h, w = initial_frame.shape[:2]

    # Load resources and initialize game/tracker
    high_score = load_high_score()
    game = AnimatedBallGame(w, h, high_score)
    tracker = HandTracker()

    # Settings for the small hand-preview window
    preview_width = 250
    preview_height = 200
    preview_x = w - preview_width - 20
    preview_y = h - preview_height - 20

    print("=" * 60)
    print("ANIMATED HAND-CONTROLLED BALL GAME (Static Background)")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Show your hand to start the game")
    print("2. Point with your INDEX FINGER to control the ball")
    print("3. Hit the glowing targets to score points!")
    print("4. Check the small box at bottom-right to see your hand tracking status.")
    print("5. Press R to reset, Q to quit, SPACE to pause")
    print("\nTip: Ball ONLY moves when your index finger is clearly visible!")
    print("=" * 60)
    
    paused = False

    while True:
        ret, camera_frame = cap.read()
        if not ret:
            # If camera disconnects, break or handle gracefully
            print("Camera feed ended.")
            break
            
        camera_frame = cv2.flip(camera_frame, 1) # Mirror the camera frame

        # --- CORE MODIFICATION: Create the static background game canvas ---
        game_frame = np.zeros((h, w, 3), dtype=np.uint8)
        game_frame[:] = (5, 5, 30) # Dark Space Blue/Purple (B, G, R)

        hand_preview = camera_frame.copy() # Base image for the preview box
        
        # 1. Hand and Face Detection (on the live camera feed)
        faces = tracker.detect_faces(camera_frame)
        center, bbox, contour, fingertips = tracker.find_hand_details(camera_frame, faces)
        
        finger_pos = None
        hand_preview_with_tracking = None

        if center is not None and not paused and not game.game_over:
            # Start game on first successful detection
            if not game.game_started:
                game.game_started = True
                game.start_timer()
            
            if fingertips and len(fingertips) > 0:
                finger_pos = fingertips[0]
                
                # --- Draw tracking overlay on the PREVIEW image only ---
                hand_preview_copy = hand_preview.copy()
                cv2.circle(hand_preview_copy, finger_pos, 15, (0, 255, 255), 3)
                cv2.circle(hand_preview_copy, finger_pos, 8, (255, 255, 0), -1)
                if contour is not None:
                    cv2.drawContours(hand_preview_copy, [contour], -1, (0, 255, 0), 2)
                if bbox is not None:
                    x, y, w_box, h_box = bbox
                    cv2.rectangle(hand_preview_copy, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(hand_preview_copy, "INDEX FINGER", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                hand_preview_with_tracking = hand_preview_copy
                # --- End Draw on Preview ---
        
        # 2. Draw face detection boxes on the PREVIEW frame
        for (fx, fy, fw, fh) in faces:
             cv2.rectangle(hand_preview, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
             cv2.putText(hand_preview, "Face Excluded", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 3. Game Logic Update
        if not paused and not game.game_over:
            game.update(finger_pos)
            
        # 4. Draw Game onto the Static Background
        final_display_frame = game.draw(game_frame)

        # 5. Draw Hand Preview Window onto the Final Display
        border_thickness = 3
        # Background box for the preview
        cv2.rectangle(final_display_frame, 
                      (preview_x - border_thickness, preview_y - border_thickness),
                      (preview_x + preview_width + border_thickness, preview_y + preview_height + border_thickness),
                      (100, 100, 200), -1) # Purple-grey border background

        # Content of the preview window
        if hand_preview_with_tracking is not None:
            # Show the tracked hand
            preview_resized = cv2.resize(hand_preview_with_tracking, (preview_width, preview_height))
            final_display_frame[preview_y:preview_y + preview_height, preview_x:preview_x + preview_width] = preview_resized
        else:
            # Show the raw camera feed when tracking is not active
            preview_resized = cv2.resize(hand_preview, (preview_width, preview_height))
            final_display_frame[preview_y:preview_y + preview_height, preview_x:preview_x + preview_width] = preview_resized
            cv2.putText(final_display_frame, "Show Your Hand", 
                        (preview_x + 30, preview_y + preview_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2) # Orange message

        # Preview title/label
        cv2.putText(final_display_frame, "HAND CONTROL (Live Feed)", (preview_x - 10, preview_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Border around the live feed
        cv2.rectangle(final_display_frame, (preview_x, preview_y),
                      (preview_x + preview_width, preview_y + preview_height),
                      (0, 255, 255), border_thickness)

        # 6. Draw Warnings
        if game.game_started and not game.game_over and finger_pos is None:
            warning_text = "Show your INDEX FINGER to move the ball!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            x_warn = (w - text_size[0]) // 2
            cv2.putText(final_display_frame, warning_text, (x_warn, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        # 7. Draw Pause Overlay
        if paused:
            overlay = final_display_frame.copy()
            cv2.rectangle(overlay, (w//2 - 150, h//2 - 50), (w//2 + 150, h//2 + 50), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.8, final_display_frame, 0.2, 0, final_display_frame)
            cv2.putText(final_display_frame, "PAUSED", (w//2 - 80, h//2 + 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)

        # Show the final display frame
        cv2.imshow("Animated Ball Game", final_display_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset game but keep high score
            high_score = game.high_score
            game = AnimatedBallGame(w, h, high_score)
            paused = False
        elif key == ord(' '):
            paused = not paused

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
