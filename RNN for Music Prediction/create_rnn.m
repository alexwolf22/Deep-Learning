function net = create_rnn(netspec_opts)
% create a RNN network based on the specfication provided by netspec_opts, which
% is expected to have 'custom' layers. A custom layer should have a
% 'subtype' field with one of the following options:
%  - recurrent_tanh
%    - uses @recurrent_tanh_forward and @recurrent_tanh_backward, which you
%    must implement. Must provide initial weights in layer.weights{1} and
%    initial biases in layer.weights{2}
%  - recurrent_lstm
%    - uses @recurrent_lstm_forward and @recurrent_lstm_backward, which you
%    must implement. Must provide initial weights in layer.weights{1} and
%    initial biases in layer.weights{2}
%  - fully_connected
%    - uses @fully_connected_forward and @fully_connected_backward, which we
%    provde. Must provide initial weights in layer.weights{1} and initial
%    biases in layer.weights{2}
%  - tanH
%    - uses @generalized_logistic_forward and @generalized_backward, which we
%    provde.
%  - sigmoid
%    - uses @generalized_logistic_forward and @generalized_backward, which we
%    provde.

net.layers = cell(1, length(netspec_opts.layers));

inputSize= netspec_opts.input_features;

% for each layer specified in netspec_opts.hidden_units
for i = 1:length(netspec_opts.layers)
  if strcmp(netspec_opts.layers(i).type, 'custom')
    % create a custom layer
    net.layers{i}.type = 'custom';
    subtype = netspec_opts.layers(i).subtype;
    net.layers{i}.subtype = subtype;
    if strcmp(subtype, 'recurrent_lstm')
      % attach the correct function handles and intialize the weights and
      % biases for the layer
      % FILL IN
      net.layers{i}.forward= @recurrent_lstm_forward;
      net.layers{i}.backward= @recurrent_lstm_backward;
      
      nOut= netspec_opts.layers(i).hidden_units;
      net.layers{i}.hidden_units= nOut;
      
      if i==1
          nIn= inputSize;
      else
          nIn= netspec_opts.layers(i-1).hidden_units;
      end
      
      wieghts= -.1 + (.1+.1)*rand(4*nOut,1, 1, nOut+nIn);
      net.layers{i}.weights{1}= wieghts;
      
      bias= zeros(4*nOut,1);
      bias(1:nOut, :)= 1;
      net.layers{i}.weights{2}= bias;


    elseif strcmp(subtype, 'recurrent_tanh')
      % attach the correct function handles and intialize the weights and
      % biases for the layer
      % FILL IN
      net.layers{i}.forward= @recurrent_tanh_forward;
      net.layers{i}.backward= @recurrent_tanh_backward;
      
      nOut= netspec_opts.layers(i).hidden_units;
      net.layers{i}.hidden_units= nOut;
      
      if i==1
          nIn= inputSize;
      else
          nIn= netspec_opts.layers(i-1).hidden_units;
      end
      
      wieghts= -.1 + (.1+.1)*rand(nOut,1, 1, nOut+nIn);
      net.layers{i}.weights{1}= wieghts;
      
      bias= zeros(nOut,1);
      net.layers{i}.weights{2}= bias;

    elseif strcmp(subtype, 'fully_connected')
      % attach the correct function handles and intialize the weights and
      % biases for the layer
      % FILL IN
      net.layers{i}.forward= @fully_connected_forward;
      net.layers{i}.backward= @fully_connected_backward;
      
      in= netspec_opts.layers(i-1).hidden_units;
      out= netspec_opts.layers(i).hidden_units;
      net.layers{i}.hidden_units= out;
      
      std = 2/sqrtm(in);
      weights = std.*randn(out, 1, 1, in);
      bias= zeros(out,1);
      
      net.layers{i}.weights{1}= weights;
      net.layers{i}.weights{2}= bias;


    elseif strcmp(subtype, 'tanH')
      % attach the correct function handles
      net.layers{i}.forward = @tanH_forward;
      net.layers{i}.backward = @tanH_backward;
    elseif strcmp(subtype, 'sigmoid')
      % attach the correct function handles
      net.layers{i}.forward = @sigmoid_forward;
      net.layers{i}.backward = @sigmoid_backward;
    end

  % elseif
  %   ...

  else
      net.layers{i}.type = netspec_opts.layers(i).type;
  end
end
