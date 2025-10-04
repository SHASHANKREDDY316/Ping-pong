import cv2
import numpy as np
import math
import time

class HandTracker:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.tip_ids = [4, 8, 12, 16, 20]
        
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        return faces
    
    def detect_hand_skin(self, frame, face_regions=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        if face_regions is not None:
            for (x, y, w, h) in face_regions:
                margin = 30
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                mask[y1:y2, x1:x2] = 0
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
        return mask
    
    def find_hand_details(self, frame, face_regions):
        mask = self.detect_hand_skin(frame, face_regions)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, None, None, None
        
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) < 3000:
            return None, None, None, None
        
        # Get convex hull for fingertip detection
        hull = cv2.convexHull(max_contour, returnPoints=True)
        
        # Find fingertips (highest points in hull)
        fingertips = []
        hull_points = hull.reshape(-1, 2)
        
        # Get topmost points as potential fingertips
        sorted_points = sorted(hull_points, key=lambda p: p[1])
        for point in sorted_points[:5]:
            fingertips.append(tuple(point))
        
        # Get hand center
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        else:
            center = None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(max_contour)
        bbox = (x, y, w, h)
        
        return center, bbox, max_contour, fingertips


class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(2, 6)
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed
        self.life = 1.0
        self.color = color
        self.size = np.random.randint(3, 8)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.3  # gravity
        self.life -= 0.02
        return self.life > 0


class AnimatedBallGame:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Ball properties
        self.ball_x = width // 2
        self.ball_y = height // 2
        self.ball_vx = 0
        self.ball_vy = 0
        self.ball_radius = 25
        self.ball_color = (255, 100, 255)
        self.ball_trail = []
        self.max_trail = 15
        
        # Game elements
        self.targets = []
        self.score = 0
        self.particles = []
        self.spawn_timer = 0
        self.game_started = False
        self.game_over = False
        
        # Timer (1 minute = 60 seconds)
        self.game_duration = 60
        self.start_time = None
        self.remaining_time = self.game_duration
        
        # Animation properties
        self.glow_pulse = 0
        self.rainbow_offset = 0
        
        # Create initial targets
        self.spawn_target()
    
    def spawn_target(self):
        if len(self.targets) < 3:
            x = np.random.randint(100, self.width - 100)
            y = np.random.randint(100, self.height - 100)
            size = np.random.randint(30, 50)
            color = self.get_rainbow_color(np.random.randint(0, 360))
            self.targets.append({'x': x, 'y': y, 'size': size, 'color': color, 'pulse': 0})
    
    def get_rainbow_color(self, hue):
        # OpenCV HSV range: H(0-179), S(0-255), V(0-255)
        hue = int(hue * 179 / 360) % 180  # Scale 0-360 to 0-179
        c = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)
        return tuple(map(int, rgb[0, 0]))
    
    def update_ball_from_finger(self, finger_pos):
        if finger_pos is None:
            # Ball stays still when no finger detected
            self.ball_vx = 0
            self.ball_vy = 0
            return
        
        fx, fy = finger_pos
        # Calculate direction from ball to finger
        dx = fx - self.ball_x
        dy = fy - self.ball_y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist > 10:
            # Move ball directly towards finger position
            speed = min(dist / 10, 15)
            self.ball_vx = (dx / dist) * speed
            self.ball_vy = (dy / dist) * speed
        else:
            # Ball reached finger position
            self.ball_vx = 0
            self.ball_vy = 0
    
    def update(self, finger_pos):
        if not self.game_started:
            return
        
        # Update ball position
        self.update_ball_from_finger(finger_pos)
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # Boundary collision with bounce
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
        
        # Update trail
        self.ball_trail.append((int(self.ball_x), int(self.ball_y)))
        if len(self.ball_trail) > self.max_trail:
            self.ball_trail.pop(0)
        
        # Check target collisions
        for target in self.targets[:]:
            dist = math.sqrt((self.ball_x - target['x'])**2 + (self.ball_y - target['y'])**2)
            if dist < self.ball_radius + target['size']:
                self.score += 10
                self.targets.remove(target)
                self.create_particles(target['x'], target['y'], target['color'], 30)
                self.spawn_target()
        
        # Spawn new targets
        self.spawn_timer += 1
        if self.spawn_timer > 120:
            self.spawn_target()
            self.spawn_timer = 0
        
        # Update targets animation
        for target in self.targets:
            target['pulse'] += 0.1
        
        # Update particles
        self.particles = [p for p in self.particles if p.update()]
        
        # Update animations
        self.glow_pulse += 0.1
        self.rainbow_offset += 2
    
    def create_particles(self, x, y, color, count=15):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))
    
    def draw_gradient_rect(self, frame, x1, y1, x2, y2, color1, color2):
        for i in range(y1, y2):
            ratio = (i - y1) / (y2 - y1)
            color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
            cv2.line(frame, (x1, i), (x2, i), color, 1)
    
    def draw(self, frame):
        # Draw animated background gradient
        self.draw_gradient_rect(frame, 0, 0, self.width, self.height, 
                                (20, 20, 40), (40, 20, 60))
        
        # Draw grid pattern
        grid_color = (50, 50, 80)
        for i in range(0, self.width, 50):
            cv2.line(frame, (i, 0), (i, self.height), grid_color, 1)
        for i in range(0, self.height, 50):
            cv2.line(frame, (0, i), (self.width, i), grid_color, 1)
        
        # Draw particles
        for particle in self.particles:
            if particle.life > 0:
                size = int(particle.size * particle.life)
                alpha = int(255 * particle.life)
                color = tuple(int(c * particle.life) for c in particle.color)
                cv2.circle(frame, (int(particle.x), int(particle.y)), size, color, -1)
        
        # Draw targets with pulsing animation
        for target in self.targets:
            pulse = math.sin(target['pulse']) * 5 + target['size']
            # Outer glow
            cv2.circle(frame, (target['x'], target['y']), int(pulse + 10), 
                      target['color'], 2)
            # Inner circle
            cv2.circle(frame, (target['x'], target['y']), int(pulse), 
                      target['color'], -1)
            # Center highlight
            cv2.circle(frame, (target['x'], target['y']), int(pulse // 3), 
                      (255, 255, 255), -1)
        
        # Draw ball trail
        for i, pos in enumerate(self.ball_trail):
            alpha = i / len(self.ball_trail)
            size = int(self.ball_radius * alpha * 0.8)
            color = tuple(int(c * alpha) for c in self.ball_color)
            cv2.circle(frame, pos, size, color, -1)
        
        # Draw ball with glow effect
        glow_size = int(self.ball_radius + abs(math.sin(self.glow_pulse)) * 10)
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), glow_size, 
                  (150, 50, 200), 3)
        
        # Main ball
        hue = (self.rainbow_offset % 360)
        ball_color = self.get_rainbow_color(hue)
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), self.ball_radius, 
                  ball_color, -1)
        
        # Ball highlight
        cv2.circle(frame, (int(self.ball_x - 8), int(self.ball_y - 8)), 8, 
                  (255, 255, 255), -1)
        
        # Draw UI elements with modern design
        self.draw_modern_ui(frame)
        
        return frame
    
    def draw_modern_ui(self, frame):
        # Score panel
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, panel_height), (20, 20, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Score text with glow
        score_text = f"SCORE: {self.score}"
        cv2.putText(frame, score_text, (18, 52), cv2.FONT_HERSHEY_DUPLEX, 1.5, 
                   (100, 50, 150), 4)
        cv2.putText(frame, score_text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, 
                   (255, 150, 255), 2)
        
        # Target count
        target_text = f"Targets: {len(self.targets)}"
        cv2.putText(frame, target_text, (self.width - 220, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 150), 2)
        
        # Instructions
        if not self.game_started:
            # Center message
            msg = "SHOW YOUR HAND TO START"
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_BOLD, 1.2, 3)[0]
            x = (self.width - text_size[0]) // 2
            y = self.height // 2
            
            # Pulsing background
            pulse = int(abs(math.sin(self.glow_pulse)) * 30) + 20
            cv2.rectangle(frame, (x - 20, y - 50), (x + text_size[0] + 20, y + 20), 
                         (pulse, pulse, pulse + 50), -1)
            
            # Text with shadow
            cv2.putText(frame, msg, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, 1.2, 
                       (0, 0, 0), 3)
            cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.2, 
                       (0, 255, 255), 3)
            
            # Sub instruction
            sub_msg = "Use your index finger to control the ball"
            text_size2 = cv2.getTextSize(sub_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            x2 = (self.width - text_size2[0]) // 2
            cv2.putText(frame, sub_msg, (x2, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (200, 200, 255), 2)
        
        # Bottom controls
        controls = "Q: Quit | R: Reset | SPACE: Pause"
        cv2.putText(frame, controls, (10, self.height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot access camera")
        return
    
    h, w = frame.shape[:2]
    
    game = AnimatedBallGame(w, h)
    tracker = HandTracker()
    
    # Hand preview box settings
    preview_width = 250
    preview_height = 200
    preview_x = w - preview_width - 20
    preview_y = h - preview_height - 20
    
    print("=" * 60)
    print("ANIMATED HAND-CONTROLLED BALL GAME")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Show your hand to start the game")
    print("2. Point with your INDEX FINGER to control the ball")
    print("3. Move your finger to move the ball")
    print("4. Hit the glowing targets to score points!")
    print("5. You have 1 MINUTE to score as many points as possible")
    print("6. Check the small box at bottom-right to see your hand")
    print("7. Press R to reset, Q to quit")
    print("\nTip: Ball ONLY moves when your index finger is visible!")
    print("=" * 60)
    
    paused = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Create a copy for hand preview
        hand_preview = frame.copy()
        
        # Detect faces
        faces = tracker.detect_faces(frame)
        
        # Find hand and fingertips
        center, bbox, contour, fingertips = tracker.find_hand_details(frame, faces)
        
        finger_pos = None
        hand_preview_with_tracking = None
        
        if center is not None and not paused:
            game.game_started = True
            
            # Use the topmost fingertip (index finger typically)
            if fingertips and len(fingertips) > 0:
                finger_pos = fingertips[0]
                
                # Draw finger tracking on PREVIEW
                hand_preview_copy = hand_preview.copy()
                cv2.circle(hand_preview_copy, finger_pos, 15, (0, 255, 255), 3)
                cv2.circle(hand_preview_copy, finger_pos, 8, (255, 255, 0), -1)
                
                # Draw hand contour on PREVIEW
                if contour is not None:
                    cv2.drawContours(hand_preview_copy, [contour], -1, (0, 255, 0), 2)
                
                if bbox is not None:
                    x, y, w_box, h_box = bbox
                    cv2.rectangle(hand_preview_copy, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    
                    # Add label on preview
                    cv2.putText(hand_preview_copy, "INDEX FINGER", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                hand_preview_with_tracking = hand_preview_copy
                
                # Draw small indicator on main frame (just a dot at finger position)
                cv2.circle(frame, finger_pos, 8, (0, 255, 255), -1)
        
        # Mark faces
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
            cv2.putText(frame, "Face", (fx, fy - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Update and draw game
        if not paused:
            game.update(finger_pos)
        frame = game.draw(frame)
        
        # Draw hand preview box at bottom-right
        # Create border and background
        border_thickness = 3
        cv2.rectangle(frame, 
                     (preview_x - border_thickness, preview_y - border_thickness),
                     (preview_x + preview_width + border_thickness, preview_y + preview_height + border_thickness),
                     (100, 100, 200), -1)
        
        # Draw preview content
        if hand_preview_with_tracking is not None:
            # Resize the hand preview to fit the box
            preview_resized = cv2.resize(hand_preview_with_tracking, (preview_width, preview_height))
            frame[preview_y:preview_y + preview_height, preview_x:preview_x + preview_width] = preview_resized
        else:
            # Show empty preview with message
            preview_resized = cv2.resize(hand_preview, (preview_width, preview_height))
            frame[preview_y:preview_y + preview_height, preview_x:preview_x + preview_width] = preview_resized
            
            # Add "No Hand Detected" text on preview
            cv2.putText(frame, "Show Your Hand", 
                       (preview_x + 30, preview_y + preview_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        # Add label above preview box
        cv2.putText(frame, "HAND CONTROL", (preview_x + 50, preview_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw border around preview
        cv2.rectangle(frame, (preview_x, preview_y),
                     (preview_x + preview_width, preview_y + preview_height),
                     (0, 255, 255), border_thickness)
        
        # Show warning when no finger detected
        if game.game_started and not game.game_over and finger_pos is None:
            warning_text = "Show your INDEX FINGER to move the ball!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            x_warn = (w - text_size[0]) // 2
            cv2.putText(frame, warning_text, (x_warn, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        
        if paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//2 - 150, h//2 - 50), 
                         (w//2 + 150, h//2 + 50), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.putText(frame, "PAUSED", (w//2 - 80, h//2 + 10), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)
        
        cv2.imshow("Animated Ball Game", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            game = AnimatedBallGame(w, h)
            paused = False
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()