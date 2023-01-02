# Regular packages are traditional packages as they existed in Python 3.2 and earlier. 
# A regular package is typically implemented as a directory containing an __init__.py file. 
# When a regular package is imported, this __init__.py file is implicitly executed, 
# and the objects it defines are bound to names in the package’s namespace. 
# The __init__.py file can contain the same Python code that any other module can contain, 
# and Python will add some additional attributes to the module when it is imported.

from effects.minimal_pipeline import MinimalPipelineEffect
from effects.xdog import XDoGEffect
from helpers.visual_parameter_def import portrait_preset, add_optional_params, remove_optional_presets, minimal_pipeline_presets, \
    minimal_pipeline_vp_ranges


xdog_params = ["blackness", "contour", "strokeWidth", "details", "saturation", "contrast", "brightness"]
minimal_pipeline_params = [x[0] for x in minimal_pipeline_vp_ranges]



def get_default_settings(name, **kwargs):
    if name == "xdog":
        effect = XDoGEffect(**kwargs)
        presets = portrait_preset
        params = xdog_params
    # 進行style transfer前
    elif name == "minimal_pipeline":
        # kwargs['enable_adapt_hue_preprocess'] = False
        # kwargs['enable_adapt_hue_postprocess'] = False

        # MinimalPipelineEffect 是 一個class
        effect = MinimalPipelineEffect()
        # 一個list，裝著各項可調slider的名稱及其默認數值的tuple
        presets = minimal_pipeline_presets
        # 各項可調slider的名稱
        params = minimal_pipeline_params
    else:
        raise ValueError(f"effect {name} not found")
    return effect, remove_optional_presets(presets, **kwargs), add_optional_params(params, **kwargs)
