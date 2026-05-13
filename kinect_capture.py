import freenect
import cv2
import numpy as np
import time

class KinectCapture:
    def __init__(self):
        self.depth_data = None
        self.rgb_data = None
        self.running = False

    def depth_callback(self, dev, data, timestamp):
        self.depth_data = data

    def video_callback(self, dev, data, timestamp):
        self.rgb_data = data

    def start(self):
        """Start capturing from Kinect"""
        self.running = True
        freenect.runloop(depth=self.depth_callback, video=self.video_callback)

    def stop(self):
        """Stop capturing"""
        self.running = False
        freenect.stop()

    def get_depth_frame(self):
        """Get the latest depth frame"""
        return self.depth_data

    def get_rgb_frame(self):
        """Get the latest RGB frame"""
        return self.rgb_data

    def get_depth_mm(self):
        """Convert depth data to millimeters"""
        if self.depth_data is None:
            return None
        # Convert from raw depth to mm
        depth_mm = freenect.depth_to_mm(self.depth_data)
        return depth_mm

def main():
    kinect = KinectCapture()

    # Start capture in a separate thread or process if needed
    import threading
    capture_thread = threading.Thread(target=kinect.start)
    capture_thread.daemon = True
    capture_thread.start()

    try:
        while True:
            depth = kinect.get_depth_mm()
            rgb = kinect.get_rgb_frame()

            if depth is not None:
                # Process depth data
                # For example, display or save
                cv2.imshow('Depth', depth.astype(np.uint8))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(0.1)  # Small delay

    except KeyboardInterrupt:
        pass
    finally:
        kinect.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\mael\Desktop\FabLab\kinect_capture.py