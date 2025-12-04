import cv2
import mediapipe as mp
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import time

class RPPGSystem:
    def __init__(self, buffer_size=300, fps=30):
        self.buffer_size = buffer_size # 300 frames = ~10 seconds at 30fps
        self.fps = fps
        self.times = []
        self.samples = []
        self.bpms = []
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # ROI Indices (Forehead approximation)
        self.forehead_indices = [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10]

        # Filter parameters
        self.low_cut = 0.8  # Hz (~48 BPM) - slightly higher to reduce low freq noise
        self.high_cut = 3.0 # Hz (~180 BPM) - reduced to focus on normal range
        
    def get_roi_mask(self, frame, landmarks):
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        points = []
        for idx in self.forehead_indices:
            pt = landmarks[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            points.append([x, y])
            
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        return mask, points

    def extract_signal(self, frame, mask):
        # Mean of RGB channels in ROI
        roi = cv2.mean(frame, mask=mask)
        return roi[0], roi[1], roi[2] # Blue, Green, Red (OpenCV uses BGR)

    def process_signal(self):
        if len(self.samples) < self.buffer_size:
            return None, 0
            
        # Calculate actual FPS from timestamps
        times = np.array(self.times[-self.buffer_size:])
        samples = np.array(self.samples[-self.buffer_size:]) # Shape: (N, 3)
        
        # Handle potential duplicate timestamps or empty arrays
        if len(times) < 2:
            return None, 0

        duration = times[-1] - times[0]
        if duration <= 0:
            return None, 0
            
        real_fps = (len(samples) - 1) / duration
        
        # POS Algorithm
        # 1. Temporal Normalization
        # Divide by mean to remove DC component and normalize
        means = np.mean(samples, axis=0)
        norm_samples = samples / means
        
        # 2. Projection
        # S1 = G - B
        # S2 = G + B - 2R
        # Note: samples are (B, G, R)
        B = norm_samples[:, 0]
        G = norm_samples[:, 1]
        R = norm_samples[:, 2]
        
        S1 = G - B
        S2 = G + B - 2*R
        
        # 3. Tuning (Alpha)
        # alpha = std(S1) / std(S2)
        std_S1 = np.std(S1)
        std_S2 = np.std(S2)
        
        if std_S2 == 0:
            alpha = 0
        else:
            alpha = std_S1 / std_S2
            
        # 4. Combination
        # P = S1 + alpha * S2
        P = S1 + alpha * S2
        
        # Resample to fixed FPS (e.g., 30Hz) for consistent filtering
        target_fps = 30.0
        num_samples = int(duration * target_fps)
        target_times = np.linspace(times[0], times[-1], num_samples)
        
        # Linear interpolation
        y_resampled = np.interp(target_times, times, P)
        
        # Detrend
        y_detrend = signal.detrend(y_resampled)
        
        # Bandpass Filter
        nyquist = 0.5 * target_fps
        low = self.low_cut / nyquist
        high = self.high_cut / nyquist
        
        # Safety check for filter bounds
        if low <= 0 or high >= 1:
            return None, 0
            
        b, a = signal.butter(3, [low, high], btype='band')
        y_filtered = signal.filtfilt(b, a, y_detrend)
        
        return y_filtered, target_fps

    def estimate_bpm(self, filtered_signal, fps):
        if filtered_signal is None or fps <= 0:
            return 0, None, None
            
        # FFT with zero padding for better resolution
        n = len(filtered_signal)
        n_padded = n * 4 # Zero padding
        freqs = fftpack.rfftfreq(n_padded, d=1.0/fps)
        fft_vals = np.abs(fftpack.rfft(filtered_signal, n=n_padded))
        
        # Filter out frequencies outside human heart rate range
        valid_idx = np.where((freqs >= self.low_cut) & (freqs <= self.high_cut))
        valid_freqs = freqs[valid_idx]
        valid_fft = fft_vals[valid_idx]
        
        if len(valid_fft) == 0:
            return 0, None, None
            
        peak_freq = valid_freqs[np.argmax(valid_fft)]
        bpm = peak_freq * 60.0
        return bpm, valid_freqs, valid_fft

    def draw_graph(self, frame, signal_data, bpm, real_fps, fft_data=None):
        h, w, _ = frame.shape
        
        # Graph area 1: Signal
        graph_h = 100
        graph_w = 300
        graph_x = w - graph_w - 20
        graph_y = h - graph_h - 20
        
        # Draw background for Signal
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
        cv2.putText(frame, "Pulse Signal", (graph_x, graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if signal_data is not None:
            # Normalize signal to fit in graph
            min_val = np.min(signal_data)
            max_val = np.max(signal_data)
            if max_val - min_val > 0:
                norm_signal = (signal_data - min_val) / (max_val - min_val)
                
                points = []
                for i, val in enumerate(norm_signal):
                    x = int(graph_x + (i / len(signal_data)) * graph_w)
                    y = int(graph_y + graph_h - (val * graph_h))
                    points.append((x, y))
                    
                # Draw signal
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (0, 255, 0), 1)

        # Graph area 2: FFT Spectrum
        fft_graph_y = graph_y - graph_h - 40
        cv2.rectangle(frame, (graph_x, fft_graph_y), (graph_x + graph_w, fft_graph_y + graph_h), (0, 0, 0), -1)
        cv2.putText(frame, "Frequency Spectrum", (graph_x, fft_graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if fft_data is not None and fft_data[0] is not None:
            freqs, fft_vals = fft_data
            if len(freqs) > 0:
                # Normalize FFT values
                max_fft = np.max(fft_vals)
                if max_fft > 0:
                    norm_fft = fft_vals / max_fft
                    
                    points = []
                    for i, val in enumerate(norm_fft):
                        # Map frequency to x-axis (from low_cut to high_cut)
                        # freqs are already filtered to [low_cut, high_cut]
                        freq_range = self.high_cut - self.low_cut
                        if freq_range > 0:
                            rel_freq = (freqs[i] - self.low_cut) / freq_range
                            x = int(graph_x + rel_freq * graph_w)
                            y = int(fft_graph_y + graph_h - (val * graph_h))
                            points.append((x, y))
                            
                    # Draw FFT
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], (255, 255, 0), 1)
                        
                    # Draw peak
                    peak_idx = np.argmax(fft_vals)
                    peak_x = points[peak_idx][0]
                    peak_y = points[peak_idx][1]
                    cv2.circle(frame, (peak_x, peak_y), 4, (0, 0, 255), -1)
            
        # Draw BPM text
        text = f"BPM: {bpm:.1f} | FPS: {real_fps:.1f}" if bpm > 0 else f"BPM: ... | FPS: {real_fps:.1f}"
        if len(self.samples) < self.buffer_size:
             text = f"Buffering: {len(self.samples)}/{self.buffer_size} | FPS: {real_fps:.1f}"
             
        cv2.putText(frame, text, (graph_x, fft_graph_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Increased buffer size to 300 (10 seconds) for better freq resolution
    rppg = RPPGSystem(buffer_size=300, fps=30) 
    
    print("Starting rPPG... Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror the frame
        frame = cv2.flip(frame, 1)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = rppg.face_mesh.process(frame_rgb)
        
        avg_bpm = 0
        filtered_signal = None
        fft_data = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get ROI
                mask, points = rppg.get_roi_mask(frame, face_landmarks.landmark)
                
                # Draw ROI
                cv2.polylines(frame, [points], True, (255, 0, 0), 1)
                
                # Extract Signal
                # val is now (B, G, R)
                b, g, r = rppg.extract_signal(frame, mask)
                rppg.samples.append((b, g, r))
                rppg.times.append(time.time())
                
                # Maintain buffer size
                if len(rppg.samples) > rppg.buffer_size:
                    rppg.samples.pop(0)
                    rppg.times.pop(0)
                
                # Process & Estimate BPM
                if len(rppg.samples) == rppg.buffer_size:
                    filtered_signal, target_fps = rppg.process_signal()
                    bpm, freqs, fft_vals = rppg.estimate_bpm(filtered_signal, target_fps)
                    fft_data = (freqs, fft_vals)
                    
                    # Smooth BPM
                    if bpm > 0:
                        rppg.bpms.append(bpm)
                        if len(rppg.bpms) > 30: # Smooth over last 30 readings
                            rppg.bpms.pop(0)
                        avg_bpm = np.mean(rppg.bpms)
                    else:
                        avg_bpm = 0
                        
        # Calculate current real FPS for display (approximate)
        real_fps_display = 0
        if len(rppg.times) > 1:
            real_fps_display = (len(rppg.times) - 1) / (rppg.times[-1] - rppg.times[0])

        # Draw Graph (always draw, even if 0, to show buffering status)
        frame = rppg.draw_graph(frame, filtered_signal, avg_bpm, real_fps_display, fft_data)
        
        cv2.imshow('Real-time rPPG', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
