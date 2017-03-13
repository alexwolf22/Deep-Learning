function res0 = recurrent_tanh_backward(layer, res0, res1)
  % Computes the backward pass of a recurrent layer having a hyperbolic tangent
  % activation function
    
  % backprop the activation
  preact= res1.cache.pre_act;
  dzdVakxt= res0.dzdstate;
  dzdx= res1.dzdx;
  
  dzdy= dzdx+dzdVakxt;
  
  dzdPreAct= generalized_logistic(preact, -1, 1, 2, dzdy);
  
  %get the weights
  wv= layer.weights{1};
  [~,~,~, t]= size(wv);
  in= layer.hidden_units;
  w= wv(:, :, :, 1:t-in);
  v= wv(:, :, :, t-in+1:t);
  [mw, ~, ~, kw]= size(w);
  [mv, ~, ~, kv]= size(v);
  [~, ~, kp, tp]= size(dzdPreAct);
  rePreAct= reshape(dzdPreAct, [kp, tp]);
  reW= reshape(w, [mw, kw]);
  reV= reshape(v, [mv, kv]);
  
  
  %compute dState for earlier timestep
  dState= reV'*rePreAct;
  dState= reshape(dState, [1, 1, kv, tp]);
  res0.dzdstate= dState;
  
  
  %get dzdx of earlier layer
  dzdxKm= reW'*rePreAct;
  dzdxKm= reshape(dzdxKm, [1, 1, kw, tp]);
  res0.dzdx= dzdxKm;
  
  % get weight and v derivative
  h1= res0.x;
  [~,~,h1k, h1t]= size(h1);
  h2= res1.cache.state;
  [~,~,h2k, ~]= size(h2);

  x= ones(1, 1, h1k+h2k, h1t);
  x(:, :, 1:h1k, :)= h1;
  x(:, :, h1k+1:h1k+h2k, :)= h2;
  b=layer.weights{2};
  
  [~, dzdw, dzdb]= fully_connected(x, wv, b, dzdPreAct);
  
  
  %update res0 with weights
  res0.dzdw{1}= dzdw;
  res0.dzdw{2}= dzdb;

end
