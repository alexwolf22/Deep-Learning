function res0 = sigmoid_backward(~, res0, res1)
% wraps generalized_logistic (which must already be implemented) for use as a
% custom layer with vl_simplenn. Implements the sigmoid setting
% generalized_logistic
  res0.dzdx = generalized_logistic(res0.x, 0, 1, 1, res1.dzdx);
end
