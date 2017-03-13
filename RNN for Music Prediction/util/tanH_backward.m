function res0 = tanH_backward(~, res0, res1)
% wraps generalized_logistic (which must already be implemented) for use as a
% custom layer with vl_simplenn. Implements the setting for hyperbolic
% tangent of generalized_logistic
  res0.dzdx = generalized_logistic(res0.x, -1, 1, 2, res1.dzdx);
end
