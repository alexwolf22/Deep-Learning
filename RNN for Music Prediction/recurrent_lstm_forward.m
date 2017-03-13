function res1 = recurrent_lstm_forward(layer, res0, res1)
  % Computes the forward pass of a recurrent layer having an lstm activation
  % function

  % store the state and cell in res1.cache
  res1.cache.state= res1.state;
  res1.cache.cell= res1.cell;

  % compute the preactivation using res0.x and res1.state
  wv= layer.weights{1};
  bias= layer.weights{2};
  
  hkMxt= res0.x;
  hkxtM= res1.state;
  [~, ~, k1 ,b]= size(hkMxt);
  [~, ~, k2 , ~]= size(hkxtM);
  
  x= ones(1,1, k1+k2, b);
  x(:, :, 1:k1, :)= hkMxt;
  x(:, :, 1+k1:k2+k1, :)= hkxtM;
  
  akxt= fully_connected(x, wv, bias);
  
  % store the preactivation in res1.cache
  res1.cache.pre_act= akxt;

  % call lstm.m and store the activation in in res1.x and res.state and the cell
  % in res1.cell
  out= layer.hidden_units;
  f= akxt(:, :, 1:out, :);
  i= akxt(:, :, 1+out:2*out, :);
  o= akxt(:, :, 1+out*2:3*out, :);
  a= akxt(:, :, 1+out*3:4*out, :);
  
  [y, c]= lstm(f, i, o, a, res1.cell);
  res1.x= y;
  res1.state= y;
  res1.cell= c;
  
end
