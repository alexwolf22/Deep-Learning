% -------------------------------------------------------------------------
function inputs = getDagNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

[~, batchSize]= size(batch);

newImages= zeros(30,30,3,batchSize);

%perform illimuniation 1/3 of the time
%  if rand > .666
%      for t=1:batchSize
%          RGBgammas= rand(1, 3)*.1;
%          images(:,:,:, t)=imadjust(images(:,:,:, t),[], [], RGBgammas);
%      end
% end

%flip images half the time
if rand > 0.5, images=fliplr(images) ; end

%crop half of the images at random sizezes
% cropBatch= batchSize/2;
% if rand >.5
%     start=1;
% else
%     start= cropBatch+1;
%     cropBatch=batchSize;
% end

%randomly crop each half of the images
for t=1:batchSize
    
    random= floor(rand*4);
    %make top left corner
    if random<1
        x=1;
        y=1;
    %bottom left
    elseif random<2
        x=1;
        y=3;
    %top right
    elseif random<3
        x=3;
        y=1;
    %bottom right
    else
        x=3;
        y=3;
    end

    xLength= 29;
    yLength= 29;
    
    
    %crop image
    endX=x+xLength;
    endY=y+yLength;
    newImages(:,:,:,t)= images(x:endX, y:endY, :, t);
    
    
end
images= single(newImages);

% end
inputs = {'input', images, 'label', labels} ;