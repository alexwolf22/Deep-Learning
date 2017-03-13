function res1 = recurrent_tanh_forward(layer, res0, res1)
  % Computes the forward pass of a recurrent layer having a hyperbolic tangent
  % activation function

  % store the state in res1.cache.state
  res1.cache.state= res1.state;

  % compute the preactivation using res0.x and res1.state
  hkMxt= res0.x;
  hkxtM= res1.state;
  [~, ~, k1 ,b]= size(hkMxt);
  [~, ~, k2 , ~]= size(hkxtM);
  
  %get data ready for fullyconnected
  x= ones(1,1, k1+k2, b);
  x(:, :, 1:k1, :)= hkMxt;
  x(:, :, 1+k1:k2+k1, :)= hkxtM;
  
  wv= layer.weights{1};
  b= layer.weights{2};
  
  akxt= fully_connected(x, wv, b);
  
  % store the preactivation in res1.cache
  res1.cache.pre_act= akxt;
  
  % compute the tanh activation and store in res1.x and res.state
  hkxt= generalized_logistic(akxt, -1, 1, 2);
  res1.x=hkxt;
  res1.state=hkxt;

end
