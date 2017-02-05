% -------------------------------------------------------------------------
function database = create_database(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
% Assign the fields 
% 1- database.images.data
% 2- database.images.labels
% 3- database.images.set
% 4- database.meta.sets
% 5- database.meta.classes

% YOUR CODE COMES HERE

% 1- Enter code to read dataset from file
datapath= strcat(opts.dataDir, '/image_categorization_dataset.mat');
data= load(datapath);
trainValD= single(data.data_tr);
sets= data.sets_tr;
labels= data.label_tr;
testD= data.data_te;

% 2- Enter code to process dataset read into required format.
[h, w, k ,t1]= size(trainValD);
[~, ~, ~ ,t2]= size(testD);

numFeats= h*w*k;

trainValD= reshape(trainValD, [numFeats, t1]);

%get only train examples for mean
trainIndexs= sets==1;
trainIndexsfull= single(ones(numFeats, t1));
trainIndexsfull= trainIndexsfull .* trainIndexs;

Logical=logical(trainIndexsfull);
trainD= trainValD(Logical);
trainD= reshape(trainD, [], sum(trainIndexs));

%find and subtract means for traina nd val data
means= mean(trainD, 2);
trainValD= trainValD- means;

means= reshape(means, h, w, k);
trainValD= reshape(trainValD, h, w, k, t1);

%store data so it can be parsed by matconvnet
totalEx= t1+t2;

alldata= zeros(h, w, k, totalEx);
alldata(1:h, 1:w, 1:k, 1:t1)= trainValD;
alldata(1:h, 1:w, 1:k, t1+1:totalEx)= testD;

label= zeros(1, totalEx);
label(1:t1)= labels;

set= zeros(1, totalEx);
set(1:t1)= sets;
set(t1+1:totalEx)=3;

images= struct('data', alldata, 'labels', label, 'set', set);
meta= struct('sets', {'train', 'val', 'test'}, 'classes', {data.clNames});
norm= struct('avgImg', means, 'std_dev', [], 'D', [], 'V', []);
database= struct('images', images, 'meta', meta, 'normalization', norm);

% Please do not modify the code below

[row_size, col_size, num_channels, total_examples] = size( database.images.data ); 

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
    z = reshape(database.images.data,[],total_examples) ;
    % SAseVE
    n = std(z,0,1) ;
    database.normalization.std_dev = n;

    z = bsxfun(@times, z, mean(n) ./ max(n, row_size)) ;
    database.images.data = reshape(z, row_size, col_size, num_channels, []) ;
end


if opts.whitenData
    z = reshape(database.images.data,[],total_examples) ;
    W = z(:,set == 1)*z(:,set == 1)'/total_examples ;
    [V,D] = eig(W) ;
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D) ;
    en = sqrt(mean(d2)) ;
    
    database.normalization.D = D;
    database.normalization.V = V;
    
    z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
    database.images.data = reshape(z, row_size, col_size, num_channels, []) ;
end

database.images.data=single(database.images.data);


