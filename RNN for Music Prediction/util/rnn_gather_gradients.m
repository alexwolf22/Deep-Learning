function res = rnn_gather_gradients(net, res)
  
for i = 1:numel(net.layers)
  for j = 2:numel(res)
    if strcmp(net.layers{i}.type, 'custom')
      if any(strcmp(net.layers{i}.subtype, {'recurrent_tanh', ...
          'recurrent_lstm', 'fully_connected'}))
        res{1}(i).dzdw{1} = res{1}(i).dzdw{1} + res{j}(i).dzdw{1};
        res{1}(i).dzdw{2} = res{1}(i).dzdw{2} + res{j}(i).dzdw{2};
      end
    end
  end
end
res = res{1};
end