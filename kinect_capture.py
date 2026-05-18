import freenect
import cv2
import numpy as np
import time
import threading

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
        for name in ('stop', 'sync_stop', 'shutdown', 'abort'):
            stop_fn = getattr(freenect, name, None)
            if callable(stop_fn):
                stop_fn()
                break

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
        try:
            return freenect.depth_to_mm(self.depth_data)
        except AttributeError:
            # Some freenect wrappers expose raw depth directly.
            return self.depth_data

def main():
    kinect = KinectCapture()

    # Start capture in a separate thread or process if needed
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
    main()