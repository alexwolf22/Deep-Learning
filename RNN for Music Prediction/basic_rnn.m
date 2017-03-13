%%%%% do not remove these lines %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1234);                                                                     %
if ~exist('feed', 'var')                                                       %
  load(fullfile('data', 'train_val_MIDIFeeder.mat'), 'feed');                      %
end                                                                            %
% How many sequence steps in an observation                                    %
seq_len = 128;                                                                 %
                                                                               %
netspec_opts = struct();                                                       %
netspec_opts.seq_len = seq_len;                                                %
netspec_opts.input_features = 75;                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Specify the type and subtype of each layer. Acceptable values of subtype
% for custom layers are 'fully_connected', 'tanH', 'sigmoid', 'recurrent_tanh',
% and 'recurrent_lstm'
%%%%%% fill this in:
% netspec_opts.layers(1).type = 'custom';
% netspec_opts.layers(1).subtype = ...
% ...
% ...
% ...
% ...
netspec_opts.layers(1).type = 'custom';
netspec_opts.layers(1).subtype= 'recurrent_tanh';
netspec_opts.layers(1).hidden_units= 128;

netspec_opts.layers(2).type = 'custom';
netspec_opts.layers(2).subtype= 'fully_connected';
netspec_opts.layers(2).hidden_units= 75;

netspec_opts.layers(3).type = 'softmaxloss';

% Build the net
net = create_rnn(netspec_opts);

opts = struct();
% Save reults to
opts.expDir = fullfile('results', sprintf('basic_rnn_%d', ...
  netspec_opts.seq_len));
% Train from beginning every time
opts.continue = false ;
% Specfy training options: learning rate, batch size, etc.
%%%%%% fill this in:
opts.learningRate=  [.001 * ones(1, 15), .0001 * ones(1, 85)];
opts.numEpochs= 100;
opts.batchSize= 32;
opts.weightDecay = 0.0005;
opts.momentum = 0.9;
% Keep track of error in additon to loss function
opts.errorLabels = {'err'};

%%%%% do not remove these lines %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define getBatchFn and start training                                         %
getBatchFn = @(x,y) feed.get_batch(x, y, netspec_opts.seq_len);                %
[net, stats] = rnn_train(net, feed.imdb, getBatchFn, opts);                    %
                                                                               %
% choose a random song from the validation set                                 %
ix = randsample(find(feed.imdb.images.set==2), 1);                             %
% use the first steps as a seed                                                %
seed = feed.encoded{ix}(:,1:netspec_opts.seq_len,:,:);                         %
% generate the next steps                                                      %
generated = rnn_generate(net, seed, netspec_opts.seq_len*4, false);            %
% write to an audio file                                                       %
rnn_write_generated_to_audio(generated, feed, ...                              %
  fullfile(opts.expDir, 'basic_rnn.wav'));                                     %
                                                                               %
basic_rnn_train_error = rnn_eval(net, ...                                      %
  fullfile('data', 'train_val_MIDIFeeder'), ...                                %
  'train', netspec_opts.seq_len);                                              %
basic_rnn_val_error = rnn_eval(net, ...                                        %
  fullfile('data', 'train_val_MIDIFeeder'), ...                                %
  'val', netspec_opts.seq_len);                                                %
save(fullfile(opts.expDir, 'basic_rnn_results'), ...                           %
  'basic_rnn_train_error', 'basic_rnn_val_error', ...                          %
  'net', 'netspec_opts', 'opts');                                              %
fprintf('train error: %.4f\t validation error: %.4f\n', ...                    %
  basic_rnn_train_error, basic_rnn_val_error);                                 %
copyfile(fullfile(opts.expDir, 'net-epoch-100.mat'), 'basic_rnn_100.mat')      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
