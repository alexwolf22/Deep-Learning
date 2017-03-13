function res = rnn_init_forward_states(net, batch_size, res0)
n = numel(net.layers);
res = struct(...
'x', cell(1,n+1), ...
'dzdx', cell(1,n+1), ...
'dzdw', cell(1,n+1), ...
'aux', cell(1,n+1), ...
'cache', cell(1,n+1), ...
'state', cell(1,n+1), ...
'stats', cell(1,n+1), ...
'time', num2cell(zeros(1,n+1)), ...
'backwardTime', num2cell(zeros(1,n+1))) ;

for i = 1:n
  if strcmp(net.layers{i}.type, 'custom')
    if nargin > 2
      % transfer the states to the next time step
      if strcmp(net.layers{i}.subtype, 'recurrent_tanh')
        res(i+1).state = res0(i+1).state;
      elseif strcmp(net.layers{i}.subtype, 'recurrent_lstm')
        res(i+1).state = res0(i+1).state;
        res(i+1).cell = res0(i+1).cell;
      end
    else
      nout = size(net.layers{i}.weights{2}, 1);
      if strcmp(net.layers{i}.subtype, 'recurrent_tanh')
        sz = [1, 1, nout, batch_size];
        res(i+1).state = zeros(sz);        
      elseif strcmp(net.layers{i}.subtype, 'recurrent_lstm')
        sz = [1, 1, nout/4, batch_size];
        res(i+1).state = zeros(sz);
        res(i+1).cell = zeros(sz);
      end
    end
  end
end