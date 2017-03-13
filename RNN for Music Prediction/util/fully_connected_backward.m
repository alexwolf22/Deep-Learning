function res0 = fully_connected_backward(layer, res0, res1)
% wraps fully_connected (which must already be implemented) for use as a
% custom layer with vl_simplenn
  [res0.dzdx, res0.dzdw{1}, res0.dzdw{2}] = ...
    fully_connected(res0.x, layer.weights{1}, layer.weights{2}, res1.dzdx);
end
