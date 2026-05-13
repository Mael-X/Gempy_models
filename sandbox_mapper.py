import numpy as np
import cv2
import json

class SandboxMapper:
    def __init__(self, lith_block_path='lith_block.npy', colors_path='colors.json',
                 extent_path='extent.npy', resolution_path='resolution.npy'):
        # Load model data
        self.lith_block = np.load(lith_block_path)
        with open(colors_path, 'r') as f:
            self.colors = json.load(f)
        self.extent = np.load(extent_path)
        self.resolution = np.load(resolution_path)

        # Convert colors to RGB tuples
        self.color_map = {}
        for lid, hex_color in self.colors.items():
            lid = int(lid)
            hex_color = hex_color.lstrip('#')
            self.color_map[lid] = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Model dimensions
        self.x_min, self.x_max = self.extent[0], self.extent[1]
        self.y_min, self.y_max = self.extent[2], self.extent[3]
        self.z_min, self.z_max = self.extent[4], self.extent[5]

        self.res_x, self.res_y, self.res_z = self.resolution

        # Kinect depth calibration
        # Assume depth in mm maps to z in model units
        # This needs calibration: depth_mm = a * z_model + b
        # For simplicity, assume 1mm = 1 unit, and z_min corresponds to some depth
        self.depth_offset = 0  # Calibrate this
        self.depth_scale = 1.0  # Calibrate this

    def depth_to_z(self, depth_mm):
        """Convert Kinect depth in mm to model z-coordinate"""
        return (depth_mm - self.depth_offset) / self.depth_scale

    def pixel_to_model_coords(self, u, v, depth_mm, image_width=640, image_height=480):
        """Convert image pixel coordinates to model x,y,z"""
        # Assume simple orthographic projection
        # Map image coords to model extent
        x = self.x_min + (u / image_width) * (self.x_max - self.x_min)
        y = self.y_min + (v / image_height) * (self.y_max - self.y_min)
        z = self.depth_to_z(depth_mm)
        return x, y, z

    def get_lithology_at_point(self, x, y, z):
        """Get lithology ID at model coordinates"""
        # Clamp to extent
        x = np.clip(x, self.x_min, self.x_max)
        y = np.clip(y, self.y_min, self.y_max)
        z = np.clip(z, self.z_min, self.z_max)

        # Convert to voxel indices
        i = int((x - self.x_min) / (self.x_max - self.x_min) * (self.res_x - 1))
        j = int((y - self.y_min) / (self.y_max - self.y_min) * (self.res_y - 1))
        k = int((z - self.z_min) / (self.z_max - self.z_min) * (self.res_z - 1))

        # Clamp indices
        i = np.clip(i, 0, self.res_x - 1)
        j = np.clip(j, 0, self.res_y - 1)
        k = np.clip(k, 0, self.res_z - 1)

        lith_id = int(self.lith_block[i, j, k])
        return lith_id

    def create_projection_image(self, depth_frame):
        """Create projected image based on depth frame"""
        if depth_frame is None:
            return None

        height, width = depth_frame.shape
        projection = np.zeros((height, width, 3), dtype=np.uint8)

        for v in range(height):
            for u in range(width):
                depth_mm = depth_frame[v, u]
                if depth_mm == 0 or depth_mm > 10000:  # Invalid depth
                    continue

                x, y, z = self.pixel_to_model_coords(u, v, depth_mm, width, height)
                lith_id = self.get_lithology_at_point(x, y, z)

                if lith_id in self.color_map:
                    projection[v, u] = self.color_map[lith_id]

        return projection

def main():
    mapper = SandboxMapper()

    # Example: create a dummy depth frame
    depth = np.random.randint(100, 1000, (480, 640), dtype=np.uint16)

    projection = mapper.create_projection_image(depth)

    if projection is not None:
        cv2.imshow('Projection', projection)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\mael\Desktop\FabLab\sandbox_mapper.py