function res = rnn_init_backward_states(net, batch_size, res, res1)
n = numel(net.layers);
for i = n:-1:1
  if strcmp(net.layers{i}.type, 'custom')
    if nargin > 3
      % transfer the state gradients to the previous time step
      if strcmp(net.layers{i}.subtype, 'recurrent_tanh')
        res(i).dzdstate = res1(i).dzdstate;
      elseif strcmp(net.layers{i}.subtype, 'recurrent_lstm')
        res(i).dzdstate = res1(i).dzdstate;
        res(i).dzdcell = res1(i).dzdcell;
      end
    else
      nout = size(net.layers{i}.weights{2}, 1);
      if strcmp(net.layers{i}.subtype, 'recurrent_tanh')
        sz = [1, 1, nout, batch_size];
        res(i).dzdstate = zeros(sz);
      elseif strcmp(net.layers{i}.subtype, 'recurrent_lstm')
        sz = [1, 1, nout/4, batch_size];
        res(i).dzdstate = zeros(sz);
        res(i).dzdcell = zeros(sz);
      end
    end
  end
end