# Usage
Experimenting with if we can use tf.keras to output both predictions and heatmap, and the resulting model can be
saved for serving.
`python grad_cam.py` would wrap a resnet to output both prediction and heatmap. `test_saved_model` would verify the
output indeed contains heatmap.

