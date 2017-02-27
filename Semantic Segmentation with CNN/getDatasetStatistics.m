function stats = getDatasetStatistics(imdb)

train = find(imdb.images.set == 1) ;

% Class statistics
classCounts = zeros(36,1) ;
for i = 1:numel(train)
  fprintf('%s: computing segmentation stats for training image %d\n', mfilename, i) ;
  lb = imdb.images.label(:,:,i) ;
  ok = lb < 255 ;
  classCounts = classCounts + accumarray(lb(ok(:))+1, 1, [36 1]) ;
end
stats.classCounts = classCounts ;

% Image statistics
rgbm1 = cell(numel(train));
rgbm2 = cell(numel(train));
for t=1:numel(train)
  fprintf('%s: computing RGB stats for training image %d\n', mfilename, t) ;
  rgb = imdb.images.data(:,:,:,t) ;
  rgb = single(rgb) ;
  z = reshape(permute(rgb,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  rgbm1{t} = sum(z,2)/n ;
  rgbm2{t} = z*z'/n ;
end
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;

stats.rgbMean = rgbm1 ;
stats.rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
