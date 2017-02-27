function ims = getBatchTest(images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [32, 32];
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.rgbMean = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.labelStride = 1 ;
opts.labelOffset = 1 ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.useGpu = false ;
opts = vl_argparse(opts, varargin);

if opts.prefetch
  % to be implemented
  ims = [] ;
  labels = [] ;
  return ;
end

if ~isempty(opts.rgbVariance) && isempty(opts.rgbMean)
  opts.rgbMean = single([128;128;128]) ;
end
if ~isempty(opts.rgbMean)
  opts.rgbMean = reshape(opts.rgbMean, [1 1 3]) ;
end

% space for images
ims = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
  size(images,4)*opts.numAugments, 'single') ;


for i=1:size(images,4)

  rgb = single(images(:,:,:,i)) ;

  for ai = 1:opts.numAugments
      
    if ~isempty(opts.rgbMean)
      ims(:,:,:,i) = bsxfun(@minus, rgb, opts.rgbMean) ;
    else
      ims(:,:,:,i) = rgb;
    end
  end
end

