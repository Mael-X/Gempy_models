import numpy as np
import cv2
import json
import threading

class SandboxMapper:
    def __init__(self, lith_block_path='lith_block.npy', colors_path='colors.json',
                 extent_path='extent.npy', resolution_path='resolution.npy'):
        # Load model data
        self.lith_block = np.load(lith_block_path)
        with open(colors_path, 'r') as f:
            self.colors = json.load(f)
        self.extent = np.load(extent_path)
        self.resolution = np.load(resolution_path)

        # Reshape flattened lithology block to 3D if needed
        self.res_x, self.res_y, self.res_z = map(int, self.resolution)
        if self.lith_block.ndim == 1 and self.lith_block.size == self.res_x * self.res_y * self.res_z:
            self.lith_block = self.lith_block.reshape((self.res_x, self.res_y, self.res_z))

        # Convert colors to RGB tuples
        self.color_map = {}
        for lid, hex_color in self.colors.items():
            lid = int(lid)
            hex_color = hex_color.lstrip('#')
            self.color_map[lid] = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        self.color_lut = self._build_color_lut()
        self.roi = None

        # Model dimensions
        self.x_min, self.x_max = self.extent[0], self.extent[1]
        self.y_min, self.y_max = self.extent[2], self.extent[3]
        self.z_min, self.z_max = self.extent[4], self.extent[5]

        # Kinect depth calibration
        # Map raw Kinect depth values in millimeters to model z-coordinates.
        # z_model = (depth_mm - depth_z_offset_mm) / depth_z_scale
        # - depth_z_offset_mm is the raw depth reading that should correspond to model z = 0.
        #   For example, if a flat surface at z = 0 is seen at 2000 mm, set offset = 2000.
        # - depth_z_scale converts millimeters into model z units.
        #   Use 1.0 if model units are millimeters, or 1000.0 if model units are meters.
        # After calibration, raw depth values are translated and scaled into the model coordinate system.
        self.depth_z_offset_mm = 1400  # measured raw depth for model z = 0
        self.depth_z_scale = 1.0       # mm per model z unit

    def depth_to_z(self, depth_mm):
        """Convert Kinect depth in mm to model z-coordinate.

        z_model = (depth_mm - depth_z_offset_mm) / depth_z_scale
        """
        return (depth_mm - self.depth_z_offset_mm) / self.depth_z_scale

    def pixel_to_model_coords(self, u, v, depth_mm, image_width=640, image_height=480):
        """Convert image pixel coordinates to model x,y,z"""
        # Assume simple orthographic projection
        # Map image coords to model extent
        x = self.x_min + (u / image_width) * (self.x_max - self.x_min)
        y = self.y_min + (v / image_height) * (self.y_max - self.y_min)
        z = self.depth_to_z(depth_mm)
        return x, y, z

    def _build_color_lut(self):
        max_id = max(self.color_map.keys()) if self.color_map else 0
        lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
        for lid, color in self.color_map.items():
            if lid >= 0:
                lut[lid] = color
        return lut

    def set_roi(self, roi):
        """Set an image ROI in pixel coordinates: (x0, y0, x1, y1)."""
        if roi is None:
            self.roi = None
            return
        x0, y0, x1, y1 = map(int, roi)
        self.roi = (max(0, x0), max(0, y0), x1, y1)

    def clear_roi(self):
        self.roi = None

    def get_roi(self):
        return self.roi

    def _get_roi_bounds(self, roi, image_width, image_height):
        if roi is None:
            roi = self.roi
        if roi is None:
            return 0, 0, image_width, image_height
        x0, y0, x1, y1 = roi
        x0 = int(np.clip(x0, 0, image_width))
        x1 = int(np.clip(x1, 0, image_width))
        y0 = int(np.clip(y0, 0, image_height))
        y1 = int(np.clip(y1, 0, image_height))
        if x1 <= x0 or y1 <= y0:
            return 0, 0, 0, 0
        return x0, y0, x1, y1

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

    def create_projection_image(self, depth_frame, roi=None, max_valid_depth=10000):
        """Create projected image based on depth frame.

        The projection is restricted to an optional ROI to reduce latency.
        """
        if depth_frame is None:
            return None

        height, width = depth_frame.shape
        projection = np.zeros((height, width, 3), dtype=np.uint8)

        x0, y0, x1, y1 = self._get_roi_bounds(roi, width, height)
        if x0 >= x1 or y0 >= y1:
            return projection

        depth_crop = depth_frame[y0:y1, x0:x1].astype(np.float32)
        valid_mask = (depth_crop > 0) & (depth_crop <= max_valid_depth)
        if not np.any(valid_mask):
            return projection

        u = np.arange(x0, x1, dtype=np.float32)
        v = np.arange(y0, y1, dtype=np.float32)
        xs = self.x_min + (u / width) * (self.x_max - self.x_min)
        ys = self.y_min + (v / height) * (self.y_max - self.y_min)
        X, Y = np.meshgrid(xs, ys)
        Z = self.depth_to_z(depth_crop)

        X = np.clip(X, self.x_min, self.x_max)
        Y = np.clip(Y, self.y_min, self.y_max)
        Z = np.clip(Z, self.z_min, self.z_max)

        i_idx = np.clip(((X - self.x_min) / (self.x_max - self.x_min) * (self.res_x - 1)).astype(np.int32), 0, self.res_x - 1)
        j_idx = np.clip(((Y - self.y_min) / (self.y_max - self.y_min) * (self.res_y - 1)).astype(np.int32), 0, self.res_y - 1)
        k_idx = np.clip(((Z - self.z_min) / (self.z_max - self.z_min) * (self.res_z - 1)).astype(np.int32), 0, self.res_z - 1)

        lith_ids = self.lith_block[i_idx, j_idx, k_idx]
        lith_ids = lith_ids.astype(np.int32, copy=False)

        lut = self.color_lut
        if lith_ids.size > 0 and lith_ids.max() >= lut.shape[0]:
            extra = np.zeros((int(lith_ids.max()) - lut.shape[0] + 1, 3), dtype=np.uint8)
            lut = np.vstack((lut, extra))

        projection_roi = lut[lith_ids]
        projection_roi[~valid_mask] = 0
        projection[y0:y1, x0:x1] = projection_roi

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
    main()