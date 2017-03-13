function res1 = tanH_forward(~, res0, res1)
% wraps generalized_logistic (which must already be implemented) for use as a
% custom layer with vl_simplenn. Implements the setting for hyperbolic
% tangent of generalized_logistic
  res1.x = generalized_logistic(res0.x, -1, 1, 2);
end
