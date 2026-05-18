import os

os.environ["DEFAULT_BACKEND"] = "PYTORCH"

import gempy as gp # type: ignore
import gempy_viewer as gpv # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # type: ignore
#Ignore car libraries bien importées !

# print("gempy version: ", gp.__version__)
# print("gempy_viewer version: ", gpv.__version__)

tabular_model = gp.create_geomodel(
    project_name="TabularModel",
    extent=[0, 1000, 0, 1000, 0, 1000],
    refinement = 8,
    structural_frame = gp.data.StructuralFrame.initialize_default_structure()
) 

tabular_model.structural_frame.structural_groups[0].name = "Stratigraphic. Units"


tabular_model.structural_frame.structural_elements[0].name = "Thithonic Layer"
tabular_model.structural_frame.structural_elements[0].name = "Berriasian Layer"
tabular_model.structural_frame.structural_elements[0].name = "Valanginian Layer"
tabular_model.structural_frame.structural_elements[0].name = "Kimmeridgian Layer"

tabular_model.structural_frame.structural_elements[0].color = '#00FFFF' # Bright Teal
tabular_model.structural_frame.structural_elements[0].color = '#006400' # Darker Green
tabular_model.structural_frame.structural_elements[0].color = '#008000' # Slight Dark Green
tabular_model.structural_frame.structural_elements[0].color = '#00FF00' # Green 


#matplotlib figure and axis the same size as the model extent, cannot get with tabular_model.extent[i] unfortunatly
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
plt.show()

tabular_model.structural_frame.structural_elements
