"""Model handling utilities.

Loads the pre-trained ML predictor (stored as a dill file) and provides a
simple inference wrapper.
"""

from __future__ import annotations

import dill
import numpy as np
import streamlit as st
from typing import Any
from .tracer.modifiers import bkg_removal, flatten
# Path to the mock model (replace with real path in production)
MODEL_PATH = "utils/mock_predictor.dill"

# Adjusting PATH to make sutra predictors accessible as package
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent 

PRED_PATH = str(BASE_DIR / "predictors" )
# print('--------------------------------')
print(str(PRED_PATH))

predictors_dict = {
    "HGBS" : {
        'processor':f'{PRED_PATH}/HGBS-C_64v2_trial-5-WBCE-processor-v2.dill', 
        'model-weights' : f'{PRED_PATH}/HGBS_C_64v2_trial-5-WBCE-weights.h5'
    }, 
    "HiGAL" : {
        'processor':f'{PRED_PATH}/HiGAL-C_64v2_trial-5-WBCE-processor-v2.dill', 
        'model-weights' : f'{PRED_PATH}/HiGAL_C_64v2_trial-5-WBCE-weights.h5'
    }
}


 

from .tracer.predictor import filamentIdentifier as fid

# @st.cache_resource
# @st.cache_data
def load_predictor(predictor_name: str) -> Any:
    """Deserialize the predictor object from disk (cached)."""
    predictor = fid(prediction_model_name = predictors_dict[predictor_name]['processor'] , model_weights = predictors_dict[predictor_name]['model-weights'])
    return predictor

@st.cache_resource
@st.cache_data
def run_model_on_cd(image: np.ndarray, model_name: str, batch_size = None) -> np.ndarray:
    """
    Run the predictor on the input CD map.

    Parameters
    ----------
    image : np.ndarray
        2-D column-density map.
    model_name : str
        Selected model identifier (currently unused - placeholder for
        multi-model support).

    Returns
    -------
    np.ndarray
        Binary skeleton map (same shape as `image`).
    """
    predictor = load_predictor(model_name)

    skeleton = predictor.predict(image, batch_size = batch_size)
    return skeleton