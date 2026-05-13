import os
import numpy as np
import gempy as gp
import gempy_viewer as gpv

# Set backend
os.environ["DEFAULT_BACKEND"] = "PYTORCH"

def create_monoclinal_model(angle_deg=25, surface_names=None, surface_colors=None, layer_thicknesses=None, base_z_start=100.0, extent=[0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50]):
    """
    Create and compute a monoclinal GemPy model.

    Returns:
        geo_model: The computed GemPy model
        lith_block: 3D array of lithology IDs
        colors_dict: Dictionary mapping lithology IDs to colors
    """
    if surface_names is None:
        surface_names = ["Tithonic", "Berriasian", "Valanginian", "Kimmeridgian"]
    if surface_colors is None:
        surface_colors = ["#00FFFF", "#006400", "#008000", "#00FF00"]
    if layer_thicknesses is None:
        layer_thicknesses = [350.0, 200.0, 150.0, 150.0]

    monoclinal_model = gp.create_geomodel(
        project_name="MonoclinalModel",
        extent=extent,
        resolution=resolution,
        structural_frame=gp.data.StructuralFrame.initialize_default_structure()
    )

    angle_rad = np.deg2rad(angle_deg)

    if angle_deg >= 85.0:
        # Vertical layers
        x_positions = [base_z_start + sum(layer_thicknesses[:i]) for i in range(len(surface_names))]
        if max(x_positions) > 900:
            scale_factor = 900.0 / max(x_positions)
            x_positions = [x * scale_factor for x in x_positions]

        x_grid_vals = np.array(x_positions)
        y_grid = np.linspace(0.0, 1000.0, 8)
        z_grid = np.linspace(100.0, 900.0, 8)
        y_mesh, z_mesh = np.meshgrid(y_grid, z_grid)

        x = []
        y = []
        z = []
        element_names_list = []

        for idx, (name, color, x_pos) in enumerate(zip(surface_names, surface_colors, x_positions)):
            if idx == 0:
                monoclinal_model.structural_frame.structural_elements[0].name = name
                monoclinal_model.structural_frame.structural_elements[0].color = color
            else:
                element = gp.data.StructuralElement(
                    name=name,
                    color=color,
                    surface_points=gp.data.SurfacePointsTable.initialize_empty(),
                    orientations=gp.data.OrientationsTable.initialize_empty()
                )
                monoclinal_model.structural_frame.structural_groups[0].append_element(element)

            x.extend([x_pos] * (8 * 8))
            y.extend(y_mesh.flatten().tolist())
            z.extend(z_mesh.flatten().tolist())
            element_names_list.extend([name] * (8 * 8))

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        element_names = element_names_list

        gp.add_surface_points(
            geo_model=monoclinal_model,
            x=x,
            y=y,
            z=z,
            elements_names=element_names
        )

        pole_vector = np.array([1.0, 0.0, 0.0])
        number_of_surfaces = len(surface_names)

        orientation_x = np.array(x_positions * 3)
        orientation_y = np.array([250.0, 500.0, 750.0] * number_of_surfaces)
        orientation_z = np.array([500.0, 500.0, 500.0] * number_of_surfaces)
        orientation_names = np.repeat(surface_names, 3)

        gp.add_orientations(
            geo_model=monoclinal_model,
            x=orientation_x,
            y=orientation_y,
            z=orientation_z,
            elements_names=orientation_names,
            pole_vector=np.vstack([pole_vector] * len(orientation_x))
        )
    else:
        # Inclined layers
        vertical_spacings = [thickness / np.cos(angle_rad) for thickness in layer_thicknesses]
        base_zs = [base_z_start]
        for spacing in vertical_spacings[:-1]:
            base_zs.append(base_zs[-1] + spacing)

        max_z_rise = 1000.0 - max(base_zs) - 50.0
        max_horizontal_extent = max_z_rise / np.tan(angle_rad)

        x_margin = 50.0
        max_x_extent = 1000.0 - 2 * x_margin
        horizontal_extent = min(max_x_extent, max_horizontal_extent)

        x_center = 500.0
        x0 = x_center - horizontal_extent / 2.0
        x1 = x_center + horizontal_extent / 2.0

        x_grid = np.linspace(x0, x1, 8)
        y_grid = np.linspace(0.0, 1000.0, 8)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        x = x_mesh.flatten()
        y = y_mesh.flatten()
        slope = np.tan(angle_rad)

        for idx, (name, color, base_z) in enumerate(zip(surface_names, surface_colors, base_zs)):
            if idx == 0:
                monoclinal_model.structural_frame.structural_elements[0].name = name
                monoclinal_model.structural_frame.structural_elements[0].color = color
            else:
                element = gp.data.StructuralElement(
                    name=name,
                    color=color,
                    surface_points=gp.data.SurfacePointsTable.initialize_empty(),
                    orientations=gp.data.OrientationsTable.initialize_empty()
                )
                monoclinal_model.structural_frame.structural_groups[0].append_element(element)

            z = base_z + (x - x0) * slope
            element_names = [name] * len(x)
            gp.add_surface_points(
                geo_model=monoclinal_model,
                x=x,
                y=y,
                z=z,
                elements_names=element_names
            )

        pole_vector = np.array([-np.sin(angle_rad), 0.0, np.cos(angle_rad)])
        number_of_surfaces = len(surface_names)

        orientation_x = np.array([300.0, 500.0, 700.0] * number_of_surfaces)
        orientation_y = np.array([500.0, 500.0, 500.0] * number_of_surfaces)
        orientation_z = np.repeat(np.array(base_zs), 3) + (orientation_x - x0) * slope
        orientation_names = np.repeat(surface_names, 3)

        gp.add_orientations(
            geo_model=monoclinal_model,
            x=orientation_x,
            y=orientation_y,
            z=orientation_z,
            elements_names=orientation_names,
            pole_vector=np.vstack([pole_vector] * len(orientation_x))
        )

    monoclinal_model.structural_frame.structural_groups[0].name = "Stratigraphic Units"

    # Compute the model
    monoclinal_model.update_transform(gp.data.GlobalAnisotropy.NONE)
    gp.compute_model(monoclinal_model, engine_config=gp.data.GemPyEngineConfig())

    # Extract lith_block
    lith_block = monoclinal_model.solutions.raw_arrays.lith_block

    # Create colors dictionary
    colors_dict = {}
    for i, (name, color) in enumerate(zip(surface_names, surface_colors)):
        colors_dict[i + 1] = color  # Lithology IDs start from 1

    return monoclinal_model, lith_block, colors_dict, extent, resolution

if __name__ == "__main__":
    # Example usage
    model, lith_block, colors, extent, resolution = create_monoclinal_model()
    print(f"Lith block shape: {lith_block.shape}")
    print(f"Extent: {extent}")
    print(f"Resolution: {resolution}")
    print(f"Colors: {colors}")

    # Save the data for later use
    np.save('lith_block.npy', lith_block)
    np.save('extent.npy', np.array(extent))
    np.save('resolution.npy', np.array(resolution))

    import json
    with open('colors.json', 'w') as f:
        json.dump(colors, f)</content>
<parameter name="filePath">c:\Users\mael\Desktop\FabLab\model_extraction.py