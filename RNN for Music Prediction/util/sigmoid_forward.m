function res1 = sigmoid_forward(~, res0, res1)
% wraps generalized_logistic (which must already be implemented) for use as a
% custom layer with vl_simplenn. Implements the sigmoid setting
% generalized_logistic
  res1.x = generalized_logistic(res0.x, 0, 1, 1);
end
