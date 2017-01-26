function database = create_database(dataset_path, opts)

% This script was written for the course CS-78/178 HW-1
% - The script is written to allow for database creation.
% - This database will be read by the solvers in MatConvNet
% Inputs:
% 1- dataset_path
% 2- opts
% Outputs:
% 1- database : the database has fields database.images
%               and database.meta
%               The database format is as follows
%               database.images is a struct that contains 3 fields
%               1-1- data
%               1-2- labels
%               1-3- sets
%               database.meta is a struct that contains 2 fields
%               2-1- sets
%               2-2- classes.

data=load(dataset_path);
label= data.labels;
features=data.features;

isSet= isfield(data, 'sets');
if isSet
    set=data.sets;
end

mean_subtration= opts.mean_subtration;
normalization= opts.normalization;

[kInit, tInit]= size(features);
%do mean subtraction on data set if specified
if mean_subtration
    means= sum(features, 2)/tInit;
    features= bsxfun(@minus, means, features)*-1;
end

%do noramilization if specified
if normalization
    
    %loop through all features
    for k=1:kInit
        feat=features(k,:);
        n=std(feat);
        features(k,:)=features(k,:)/n;     
    end
end

%create sets if needed
if ~isSet
    set=ones(1,tInit);
end

%get other data ready for db
data= reshape(features, [1, 1, kInit, tInit]);
sets= {'train', 'val', 'test'};

%creating class nmes for xor datset
if strcmp('xor_dataset.mat', dataset_path)
    classes= cell(1,2);
    classes(1)= {'not xor'};
    classes(2)= {'xor'};
%create class name for iris dats set
else
    classes= cell(1,3);
    classes(1)= {'flower1'};
    classes(2)= {'flower2'};
    classes(3)= {'flower3'};
end


%init nested structs
images= struct('data', data, 'label', label, 'set', set);
meta= struct('sets', sets, 'classes', {classes});

database= struct('images', images, 'meta', meta);

