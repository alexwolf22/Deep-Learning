function res0 = recurrent_lstm_backward(layer, res0, res1)
  % Computes the backward pass of a recurrent layer having an lstm activation
  % function

  % restore the preactivation from res1.cache
  % i, f, o, a, c
  akxt= res1.cache.pre_act;
  out= layer.hidden_units;
  f= akxt(:, :, 1:out, :);
  i= akxt(:, :, 1+out:2*out, :);
  o= akxt(:, :, 1+out*2:3*out, :);
  a= akxt(:, :, 1+out*3:4*out, :);
  c= res1.cache.cell;
  
  dzdstate= res0.dzdstate; %maybe this should be res1
  dzdx=res1.dzdx;
  dzdy= dzdstate+dzdx;
  dzdc=res0.dzdcell;

  % backprop state and cell through the lstm.m functon
  [dzdi, dzdf, dzdo, dzda, dzdc]= lstm(f, i, o, a, c, dzdy, dzdc);

  % backprop the gradient through the preactivaton function
  [~, ~, m, bs]= size(dzdi);
  akxt= ones(1, 1, 4*m, bs);
  akxt(:, :, 1:m, :)= dzdf;
  akxt(:, :, 1+m:2*m, :)= dzdi;
  akxt(:, :, 1+2*m:3*m, :)= dzdo;
  akxt(:, :, 1+3*m:4*m, :)= dzda;
  
  %get dzdw and dzdb
  h1= res0.x;
  [~,~,h1k, h1t]= size(h1);
  h2= res1.cache.state;
  [~,~,h2k, ~]= size(h2);
  x= ones(1, 1, h1k+h2k, h1t);
  x(:, :, 1:h1k, :)= h1;
  x(:, :, h1k+1:h1k+h2k, :)= h2;
  
  wv= layer.weights{1};
  b=layer.weights{2};
  
  [~, dzdw, dzdb]= fully_connected(x, wv, b, akxt);
  
  akxt= reshape(akxt, [4*m, bs]);
  
  %get dzdx
  [~,~,~, t]= size(wv);
  w= wv(:, :, :, 1:t-out);
  [wm, ~, ~, wt]= size(w);
  w= reshape(w, wm, wt);

  dzdx= w'*akxt; 
  [s1, s2]= size(dzdx);
  dzdx= reshape(dzdx, [1, 1, s1, s2]);
  
  %get dzdstate
  v= wv(:, :, :, t-out+1:t);
  [vm, ~, ~, vt]= size(v);
  v= reshape(v, [vm, vt]);
  dzdstate= v'*akxt;
  [mstate, tstate]= size(dzdstate);
  dzdstate= reshape(dzdstate, [1, 1, mstate, tstate]);
  
  %get dzdcell
  dzdcell=dzdc.*generalized_logistic(f, 0, 1, 1);

  grads = {dzdcell, dzdw, dzdb, dzdx, dzdstate};
  res0.dzdcell = grads{1};
  res0.dzdw{1} = grads{2};
  res0.dzdw{2} = grads{3};
  res0.dzdx = grads{4};
  res0.dzdstate = grads{5};
end
