# Manual Template Assets

This directory contains SVG files that are manually created or composed
in Inkscape. These serve as base templates for code-generated figures.

## Files

- [hidden_unit.svg](hidden_unit.svg): Base neuron template with labeled
  layers for programmatic colorization. Used by
  [parameter_symmetries.py](../../../notebooks/figures/main/dissociation/parameter_symmetries.py)
  to generate the symmetry illustrations in Figure 1.

## Workflow

1. **Edit in Inkscape**: Open the SVG in Inkscape to modify structure,
   add new elements, or adjust layouts.

2. **Use Inkscape labels**: Elements are identified by their `inkscape:label`
   attribute (visible in Object > Object Properties). The code uses these
   labels to find and colorize specific elements.

3. **Preserve layer structure**: The template uses Inkscape layers to organize
   elements. Keep the layer hierarchy intact for correct rendering.

4. **Run the notebook to regenerate**: After editing the template, re-run
   `parameter_symmetries.py` to regenerate the illustrations in
   [figures/main/dissociation/symmetries/](../../main/dissociation/symmetries/).

## Label Conventions

The `hidden_unit.svg` template uses these labels for colorization:

- `w_in_1`, `w_in_2`, `w_in_3`: Incoming weights
- `bias`: Bias input
- `w_out_1`, `w_out_2`: Outgoing weights
- `soma`: Neuron body
- `outline`: Outer glow/outline
- `constant`, `elu`, `tanh`, `relu`: Activation function curves
  (visibility toggled by code)
