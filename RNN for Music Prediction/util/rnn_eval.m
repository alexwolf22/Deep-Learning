function error = rnn_eval(net, MIDIFeeder_instance_path, which_set, seq_len)
%%%%% do not remove these lines
rng(1234);
if ~exist('feed', 'var')
  load(MIDIFeeder_instance_path, 'feed');
end

switch which_set
  case 'train'
    set = 1;
  case 'val'
    set = 2;
  case 'test'
    set = 3;
end

batch = find(feed.imdb.images.set == set);
batchSize = numel(batch);
[seq_x, seq_labels] = feed.get_batch([], batch, seq_len);
res = rnn_init_forward_states(net, batchSize);
error = 0;
for i = 1:seq_len-1
  x = seq_x(:,i,:,:);
  labels = seq_labels(:,i,:,:);
  net.layers{end}.class = labels;
  res = vl_simplenn(net, x, [], res, 'Mode', 'test');
  multi_err = error_multiclass(labels, res);
  error = error + multi_err(1);
end
num = numel(find(seq_labels(:,:,2,:)~=0));
error = error / num;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -------------------------------------------------------------------------
function err = error_multiclass(labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = labels(:,:,1,:) > 0 ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

m = min(5, size(predictions,3)) ;

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ;
