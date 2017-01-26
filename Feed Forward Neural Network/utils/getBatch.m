function varargout = getBatch(imdb, batch)
% Your get batch function is responsible for data pre-processing at run
% time.
data = imdb.images.data(:,:,:,batch);
labels = imdb.images.label(1,batch);
varargout{1} = {'input', data, 'label', labels};
