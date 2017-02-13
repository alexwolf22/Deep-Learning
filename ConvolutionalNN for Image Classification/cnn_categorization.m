function [net, info] = cnn_categorization(varargin)
% CNN_ASSIGNMENT   Demonstrates MatConvNet on Image Categorization task
%    The script allows for using two models: 'Base' and 'Advanced'. Use the 'modelType' option to choose one.

rng(1234);


current_path = '/Users/alexwolf/desktop/cs78/hw2';

% Setup the matconvnet environment by calling the vl_setup file.
path_to_matconvnet = ['~/Documents/MATLAB/matconvnet'];
cd([path_to_matconvnet '/matlab/'])
vl_setupnn 
cd(current_path)

opts.modelType = 'advanced' ;
opts = vl_argparse(opts, varargin) ;

% SET IMPORTANT PATHS HERE% EXP DIR defines where the snapshots of your model are stored
opts.expDir = '/Users/alexwolf/desktop/cs78/hw2/results/advModel';
% DATA DIR defines where the data is to be loaded from
opts.dataDir = '/Users/alexwolf/desktop/cs78/hw2' ;

% DATABASE DIR defines where the database created is stored.
opts.database_path = 'database-advanced-categorization.mat';

% Options for the network
opts.networkType = 'dagnn' ;
opts.train = struct() ;
opts.train.gpus = [];
opts.whitenData= true;
opts.contrastNormalization= true;
opts = vl_argparse(opts, varargin) ;


% -------------------------------------------------------------------------
% Prepare Data                                                   
% -------------------------------------------------------------------------

% Specify the pre-processing options for creating the dataset HERE
% opts.whitenData 
% opts.contrastNormalization 


% IF the database exists, then read from it otherwise create it and save it
if exist(opts.database_path, 'file')
    database = load(opts.database_path) ;
else
    
    database = create_database(opts) ;
    mkdir(opts.expDir) ;
    save(opts.database_path, '-struct', 'database') ;
end

% -------------------------------------------------------------------------
% Prepare Model                                                   
% -------------------------------------------------------------------------

% Create a model of the type specified
switch opts.modelType    
    case 'advanced'

        % ENTER ALL YOUR CODE WITHIN THE - - - LINES BELOW OR CREATE A NEW FILE cnn_categorization_mod.m 
        % Specify the struct netspec_opts for the base model HERE
        kernel= zeros(2,16);
        kernel(:,1)= 3; %32 filt 16
        
        kernel(:,4)= 5; %32 filt 32 strdide 2 k=5
        
        kernel(:,7)= 3; %16x16 filt 64 stride 1 ,k 3
        
        kernel(:,10)= 5; %8x8 filts 128 stride 2  k=5
        
        kernel(:,13)= 5; %4x4 filts 128 stride 2  k=5
        
        kernel(:,16)= 4; %avg pool
        
        filters= zeros(1, 16);
        filters(1)= 16;
        filters(2)= 16;
        
        filters(4)= 32;
        filters(5)= 32;
        
        filters(7)= 64;
        filters(8)= 64;
        
        filters(10)= 128;
        filters(11)= 128;
        
        filters(13)= 256;
        filters(14)= 256;
        
        strides= zeros(1, 16);
        strides(1)= 1; 
        
        strides(4)= 2;
        
        strides(7)= 1;
        
        strides(10)= 2;
        
        strides(13)= 2;
        
        strides(16)= 1;
       
        layers= cell (1,16);
        layers{1}= 'conv';
        layers{2}= 'bn';
        layers{3}= 'relu';
        layers{4}= 'conv';
        layers{5}= 'bn';
        layers{6}= 'relu';
        layers{7}= 'conv';
        layers{8}= 'bn';
        layers{9}= 'relu';
        layers{10}= 'conv';
        layers{11}= 'bn';
        layers{12}= 'relu';
        layers{13}= 'conv';
        layers{14}= 'bn';
        layers{15}= 'relu';
        layers{16}= 'pool';
                       
        netspec_opts= struct('kernel_size', kernel,'num_filters', filters, 'stride', strides,'layer_type', []);
        netspec_opts.layer_type= layers;
        
        % Specift the struct train_opts for the base model HERE
        wD= .0001;
        batch= 128;
        momentum=.9;
        
%         nE= 32;
%         lR= ones(1,30)*.01;
%         lR(1:5)=.1;
%         lR(16:17)=.001;
%         lR(18:20)=.0004;

%         nE= 18;
%         lR= ones(1,30)*.01;
%         lR(1:10)=.1;
%         lR(11:15)=.001;
%         lR(16:18)=.0001;

%         nE= 15;
%         lR= ones(1,18)*.01;
%         lR(1:2)=.1;
%         lR(3:11)=.01;
%         lR(12:15)=.001;

%          nE= 15;
%          lR= ones(1,18)*.01;
%          lR(12:15)=.001;

%          nE= 25;
%          lR= ones(1,25)*.01;
%          lR(1)=.1;
%          lR(20:25)=.001;

%          nE= 25;
%          lR= ones(1,25)*.01;
%          lR(1:5)=.1;
%          lR(16:20)=.001;
%          lR(21:25)=.0001;

%          nE= 25;
%          lR= ones(1,25)*.01;
%          lR(1:15)=.1;
%          lR(21:23)=.001;
%          lR(21:25)=.0001;

        nE= 22;
        lR= ones(1,22);
        lR(1:9)=.1;
        lR(10:15)=.01;
        lR(16:19)=.001;
        lR(20:22)=.0001;
        
        train_opts= struct('learningRate', lR,'weightDecay', wD, 'batchSize', batch,'momentum', momentum,'numEpochs', nE);
        classes= database.meta.classes;
        net = cnn_categorization_advanced(netspec_opts, train_opts, classes);
    case 'base'

        % Specify the struct netspec_opts for the base model HERE
        kernel= zeros(2,10);
        kernel(:,1)= 3;
        kernel(:,4)= 3;
        kernel(:,7)= 3;
        kernel(:,10)= 8;
        
        filters= zeros(1, 10);
        filters(1)= 16;
        filters(2)= 16;
        filters(4)= 32;
        filters(5)= 32;
        filters(7)= 64;
        filters(8)= 64;
        
        strides= zeros(1, 10);
        strides(1)=1;
        strides(4)=2;
        strides(7)=2;
        strides(10)=1;
       
        layers= cell (1,10);
        layers{1}= 'conv';
        layers{2}= 'bn';
        layers{3}= 'relu';
        layers{4}= 'conv';
        layers{5}= 'bn';
        layers{6}= 'relu';
        layers{7}= 'conv';
        layers{8}= 'bn';
        layers{9}= 'relu';
        layers{10}= 'pool';
                
        netspec_opts= struct('kernel_size', kernel,'num_filters', filters, 'stride', strides,'layer_type', []);
        netspec_opts.layer_type= layers;
        
        % Specift the struct train_opts for the base model HERE
        wD= .0001;
        batch= 128;
        momentum=.9;
        
        nE= 25;
        lR= ones(1,25)*.01;
        lR(20:25)=.001;
        
        train_opts= struct('learningRate', lR,'weightDecay', wD, 'batchSize', batch,'momentum', momentum,'numEpochs', nE);
        classes= database.meta.classes;
        net = cnn_categorization_base(netspec_opts, train_opts, classes) ;
    otherwise
        error('Unknown model type ''%s''.', opts.modelType) ;
end


% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------
get_batch_function_handle = @(x,y) getDagNNBatch(x,y) ;

trainfn = @cnn_train_dag ;
[net, info] = trainfn(net, database, get_batch_function_handle, ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts,...
    'val', find(database.images.set == 2) );





