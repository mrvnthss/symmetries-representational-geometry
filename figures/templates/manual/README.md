# Manual Template Assets

This directory contains SVG files that are manually created or composed
in Inkscape. These serve as base templates for code-generated figures.

## Files

- `hidden-unit.svg`: Base neuron template with labeled layers for
  programmatic colorization. Used by `panel_d_symmetries.py` to generate
  the symmetry stack visualizations.

## Workflow

1. **Edit in Inkscape**: Open the SVG in Inkscape to modify structure,
   add new elements, or adjust layouts.

2. **Use Inkscape labels**: Elements are identified by their `inkscape:label`
   attribute (visible in Object > Object Properties). The code uses these
   labels to find and colorize specific elements.

3. **Preserve layer structure**: The template uses Inkscape layers to organize
   elements. Keep the layer hierarchy intact for correct rendering.

4. **Run notebooks to regenerate**: After editing a template, re-run the
   relevant notebook to regenerate the derived figures in `generated/`.

## Label Conventions

The `hidden-unit.svg` template uses these labels for colorization:

- `w_in_1`, `w_in_2`, `w_in_3`: Incoming weights
- `bias`: Bias input
- `w_out_1`, `w_out_2`: Outgoing weights
- `soma`: Neuron body
- `outline`: Outer glow/outline
- `constant`, `elu`, `tanh`, `relu`: Activation function curves
  (visibility toggled by code)
