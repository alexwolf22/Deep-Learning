function res1 = fully_connected_forward(layer, res0, res1)
% wraps fully_connected (which must already be implemented) for use as a
% custom layer with vl_simplenn
  res1.x = fully_connected(res0.x, layer.weights{1}, layer.weights{2});
end

