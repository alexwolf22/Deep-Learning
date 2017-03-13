function [iscorrect, err] = recurrent_tanh_test()
%%%%% Set RNG for repeatability
rng(1234);
%%%%%
netspec_opts = struct();
netspec_opts.input_features = 3;
netspec_opts.layers(1).type = 'custom';
netspec_opts.layers(1).subtype = 'recurrent_tanh';
netspec_opts.layers(1).hidden_units = 5;
netspec_opts.layers(2).type = 'custom';
netspec_opts.layers(2).subtype = 'recurrent_tanh';
netspec_opts.layers(2).hidden_units = 5;
netspec_opts.layers(3).type = 'custom';
netspec_opts.layers(3).subtype = 'fully_connected';
netspec_opts.layers(3).hidden_units = 3;
netspec_opts.layers(4).type = 'softmaxloss';

rng(9778);
net = create_rnn(netspec_opts);

nin = netspec_opts.input_features;
nout = netspec_opts.layers(end-1).hidden_units;
ts = 3;
bs = 3;
DELTA = 1e-06;
TOL = 1e-8;

iscorrect = true;
x = randn(1,ts,nin,bs);
y = randi(nout, 1, ts, 1, bs);

res = cell(1, ts);
res{1} = rnn_init_forward_states(net, bs);
res = forward(net, x, y, bs, ts, res);
res{ts} = rnn_init_backward_states(net, bs, res{ts});
res = backward(net, y, bs, ts, res);
res = rnn_gather_gradients(net, res);

dzdx_t = res(1).dzdx;
dzdw_t = res(1).dzdw{1};
dzdb_t = res(1).dzdw{2};
dzdstate_t = res(1).dzdstate;

dzdx_n = zeros(1, numel(dzdx_t));
dzdw_n = zeros(1, numel(dzdw_t));
dzdb_n = zeros(1, numel(dzdb_t));
dzdstate_n = zeros(1, numel(dzdstate_t));

res = cell(1, ts);
for j = 1:bs
  for i = 1:nin
    x_test0 = x;
    x_test0(:,1,i,j) = x_test0(:,1,i,j) + DELTA;
    res{1} = rnn_init_forward_states(net, bs);
    res = forward(net, x_test0, y, bs, ts, res);
    dzdx0 = sum(cellfun(@(c) c(end).x, res));

    x_test1 = x;
    x_test1(:,1,i,j) = x_test1(:,1,i,j) - DELTA;
    res{1} = rnn_init_forward_states(net, bs);
    res = forward(net, x_test1, y, bs, ts, res);
    dzdx1 = sum(cellfun(@(c) c(end).x, res));
    dzdx_n(:, (j-1)*nin + i) = (dzdx0(:) -  dzdx1(:)) ./ (2*DELTA);
  end
end

w_orig = net.layers{1}.weights{1};
for i = 1:numel(dzdw_t)
  w = w_orig;
  w(i) = w(i) + DELTA;
  net.layers{1}.weights{1} = w;
  res{1} = rnn_init_forward_states(net, bs);
  res = forward(net, x, y, bs, ts, res);
  dzdw0 = sum(cellfun(@(c) c(end).x, res));

  w = w_orig;
  w(i) = w(i) - DELTA;
  net.layers{1}.weights{1} = w;
  res{1} = rnn_init_forward_states(net, bs);
  res = forward(net, x, y, bs, ts, res);
  dzdw1 = sum(cellfun(@(c) c(end).x, res));
  dzdw_n(:, i) = (dzdw0(:) -  dzdw1(:)) ./ (2*DELTA);
end
net.layers{1}.weights{1} = w_orig;

b_orig = net.layers{1}.weights{2};
for i = 1:numel(dzdb_t)
  b = b_orig;
  b(i) = b(i) + DELTA;
  net.layers{1}.weights{2} = b;
  res{1} = rnn_init_forward_states(net, bs);
  res = forward(net, x, y, bs, ts, res);
  dzdb0 = sum(cellfun(@(c) c(end).x, res));

  b = b_orig;
  b(i) = b(i) - DELTA;
  net.layers{1}.weights{2} = b;
  res{1} = rnn_init_forward_states(net, bs);
  res = forward(net, x, y, bs, ts, res);
  dzdb1 = sum(cellfun(@(c) c(end).x, res));
  dzdb_n(:, i) = (dzdb0(:) -  dzdb1(:)) ./ (2*DELTA);
end
net.layers{1}.weights{2} = b_orig;

for i = 1:numel(dzdstate_t)
  res{1} = rnn_init_forward_states(net, bs);
  res{1}(2).state(i) = res{1}(2).state(i) + DELTA;
  res = forward(net, x,  y,bs, ts, res);
  dzdstate0 = sum(cellfun(@(c) c(end).x, res));

  res{1} = rnn_init_forward_states(net, bs);
  res{1}(2).state(i) = res{1}(2).state(i) - DELTA;
  res = forward(net, x, y, bs, ts, res);
  dzdstate1 = sum(cellfun(@(c) c(end).x, res));
  dzdstate_n(:, i) = (dzdstate0(:) -  dzdstate1(:)) ./ (2*DELTA);
end

dzdx_n = reshape(sum(dzdx_n, 1), size(x(:,1,:,:)));
dzdw_n = reshape(sum(dzdw_n, 1), size(dzdw_t));
dzdb_n = reshape(sum(dzdb_n, 1), size(dzdb_t));
dzdstate_n = reshape(sum(dzdstate_n, 1), size(dzdstate_t));
 
[c, e] = test(dzdx_n, dzdx_t, TOL);
iscorrect = iscorrect && c;
err.dzdx = e;

[c, e] = test(dzdw_n, dzdw_t, TOL);
iscorrect = iscorrect && c;
err.dzdw = e;

[c, e] = test(dzdb_n, dzdb_t, TOL);
iscorrect = iscorrect && c;
err.dzdb = e;

[c, e] = test(dzdstate_n, dzdstate_t, TOL);
iscorrect = iscorrect && c;
err.dzdstate = e;

function res = forward(net, x, y, bs, ts, res)
summed = 0;
for i = 1:ts
  net.layers{end}.class = y(:,i,:,:);
  if i > 1
    res{i} = rnn_init_forward_states(net, bs, res{i-1});
  end
  res{i} = vl_simplenn(net, x(:,i,:,:), [], res{i});
  summed = summed + res{i}(end).x;
end

function res = backward(net, y, bs, ts, res)
for i = ts:-1:1
  net.layers{end}.class = y(:,i,:,:);
  if i < ts
    res{i} = rnn_init_backward_states(net, bs, res{i}, res{i+1});
  end
  res{i} = vl_simplenn(net, [], 1, res{i}, 'SkipForward', true);
end   

function [isclose, err] = test(x, y, tol)
err = max(abs(x(:) - y(:)));
isclose = err < tol;
