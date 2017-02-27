function semantic_segmentation_train_mod(varargin)
% Train FCN model using MatConvNet

rng(1234);

current_path = pwd;

local_path_to_matconvnet = '~/Documents/MATLAB';
path_to_matconvnet = [local_path_to_matconvnet '/matconvnet/'];
cd([path_to_matconvnet 'matlab/'])
vl_setupnn 
cd(current_path)

% IMPORTANT PATHS ARE SPECIFIED HERE

opts.modelType = 'advanced' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = current_path;
opts.dataDir = current_path;

opts.databasePath = fullfile(opts.expDir, 'database-semantic-segmentation.mat');
opts.databaseStatsPath = fullfile(opts.expDir, 'databaseStats-semantic-segmentation.mat') ;

opts = vl_argparse(opts, varargin) ;


opts.numFetchThreads = 1 ; 
opts.train = struct() ;



% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

if exist(opts.databasePath)
  database = load(opts.databasePath) ;
else
  database = create_database(opts.dataDir);
  mkdir(opts.expDir) ;
  save(opts.databasePath, '-struct', 'database') ;
end

% Get dataset statistics
if exist(opts.databaseStatsPath)
  stats = load(opts.databaseStatsPath) ;
else
  stats = getDatasetStatistics(database) ;
  save(opts.databaseStatsPath, '-struct', 'stats') ;
end

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

% Specify your training policy here
% train_opts.batchSize
% train_opts.weightDecay
% train_opts.learningRate
% train_opts.numEpochs 
% train_opts.momentum 
% train_opts.expDir

switch opts.modelType
    
    case 'base-semantic'

      % % Specify your netspec_opts HERE
      
        % Specify the struct netspec_opts for the base model HERE
        kernel= zeros(2,17);
        kernel(:,1)= 3;
        kernel(:,4)= 3;
        kernel(:,7)= 3;
        kernel(:,10)= 3;
        kernel(:,13)= 1;
        kernel(:,14)= 4;
        kernel(:,15)= 1;
        kernel(:,17)= 2;
        
        filters= zeros(1, 17);
        filters(1)= 16;
        filters(2)= 16;
        filters(4)= 32;
        filters(5)= 32;
        filters(7)= 64;
        filters(8)= 64;
        filters(10)= 128;
        filters(11)= 128;
        filters(13)= 36;
        filters(14)= 36;
        filters(15)= 36;
        filters(17)= 36;
        
        strides= zeros(1, 17);
        strides(1)=1;
        strides(4)=2;
        strides(7)=2;
        strides(10)=2;
        strides(13)=1;
        strides(15)=1;
       
        layers= cell (1,17);
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
        layers{14}= 'convt';
        layers{15}= 'skip';
        layers{16}= 'sum';
        layers{17}= 'convt';
        
        %make inputs; only do for cells where not prev layer
        inputs= cell(2, 17);
        inputs(1, 15)= {'relu_2'};
        inputs(1, 16)= {'skip_6'};
        inputs(2, 16)= {'convt_4x'};
        
        %makes changes for skip layer
                
        netspec_opts= struct('input', [],'kernel_size', kernel,'num_filters', filters, 'stride', strides,'layer_type', []);
        netspec_opts.layer_type= layers;
        netspec_opts.input= inputs;
        
        % Specify the struct train_opts for the base model HERE
        wD= .001;
        batch= 24;
        momentum=.9;

        nE= 45;
        lR= ones(1,45)*.1;
        lR(31:45)=.01;

        train_opts= struct('learningRate', lR,'weightDecay', wD, 'batchSize', batch,'momentum', momentum,'numEpochs', nE);
        net = semantic_segmentation_base(netspec_opts, train_opts) ;

    case 'advanced'    

        % % reg conv layers
        kernel= zeros(2,35);
        kernel(:,1)= 3;
        kernel(:,4)= 3;
        kernel(:,7)= 5;
        kernel(:,10)= 3;
        kernel(:,13)= 3;
        kernel(:,16)= 5;
        kernel(:,19)= 3;
        kernel(:,22)= 3;
        kernel(:,25)= 3;
        kernel(:,28)= 1;
        
        %upsampling layers
        kernel(:,29)= 2;
        kernel(:,30)= 1;
        kernel(:,32)= 2;
        kernel(:,33)= 1;
        kernel(:,35)= 2;
        
        filters= zeros(1, 35);
        %conv layers
        filters(1)= 32;
        filters(2)= 32;
        filters(4)= 64;
        filters(5)= 64;
        filters(7)= 128;
        filters(8)= 128;
        filters(10)= 32;
        filters(11)= 32;
        filters(13)= 64;
        filters(14)= 64;
        filters(16)= 128;
        filters(17)= 128;
        filters(19)= 32;
        filters(20)= 32;
        filters(22)= 64;
        filters(23)= 64;
        filters(25)= 128;
        filters(26)= 128;
        filters(28)= 36;
        %upsamplig layers
        filters(29)= 36;
        filters(30)= 36;
        filters(32)= 36;
        filters(33)= 36;
        filters(35)= 36;
        
        strides= zeros(1, 35);
        %conv layers
        strides(1)=1;
        strides(4)=1;
        strides(7)=2;
        strides(10)=1;
        strides(13)=1;
        strides(16)=2;
        strides(19)=1;
        strides(22)=1;
        strides(25)=2;
        strides(28)=1;
        %upsampling layers
        strides(30)=1;
        strides(33)=1;
        
       
        layers= cell (1,35);
        %conv layer
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
        layers{16}= 'conv';
        layers{17}= 'bn';
        layers{18}= 'relu';
        layers{19}= 'conv';
        layers{20}= 'bn';
        layers{21}= 'relu';
        layers{22}= 'conv';
        layers{23}= 'bn';
        layers{24}= 'relu';
        layers{25}= 'conv';
        layers{26}= 'bn';
        layers{27}= 'relu';
        layers{28}= 'conv';
        %upsampling layers
        layers{29}= 'convt';
        layers{30}= 'skip';
        layers{31}= 'sum';
        layers{32}= 'convt';
        layers{33}= 'skip';
        layers{34}= 'sum';
        layers{35}= 'convt';
        
        %make inputs; only do for cells where not prev layer
        inputs= cell(2, 35);
        inputs(1, 30)= {'relu_8'};%skip_11
        inputs(1, 33)= {'relu_5'};%skip_22
        
        inputs(1, 31)= {'skip_11'};%sum_11
        inputs(2, 31)= {'convt_2x_10'};
        
        inputs(1, 34)= {'skip_12'};%sum_12
        inputs(2, 34)= {'convt_2x_11'};
        
        %makes changes for skip layer
                
        netspec_opts= struct('input', [],'kernel_size', kernel,'num_filters', filters, 'stride', strides,'layer_type', []);
        netspec_opts.layer_type= layers;
        netspec_opts.input= inputs;
        
        %training policy
        wD= .001;
        batch= 24;
        momentum=.9;

        nE= 41;
        lR= ones(1,41)*.1;
        lR(27:29)=.05;
        lR(30:34)=.01;
        lR(35:38)= .001;
        lR(39:41)= .0005;
        expDir= strcat(current_path,'/data/advModelFinal');

        train_opts= struct('expDir',expDir,'learningRate', lR,'weightDecay', wD, 'batchSize', batch,'momentum', momentum,'numEpochs', nE);
        
        
        

        net = semantic_segmentation_advanced(netspec_opts,train_opts) ;
    otherwise
        error('Unknown model type ''%s''.', opts.modelType) ;
end

net.meta.normalization.rgbMean = stats.rgbMean ;
net.meta.classes = database.classes.name ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

%make everything that was a val a train for submission
if strcmp(opts.modelType, 'advanced')
    sets= database.images.set;
    [~,width]= size(sets);
    for n=1:width
        if sets(n)==2
            sets(n)=1;
        end
    end
end
database.images.set=sets;

% Get training and test/validation subsets
train = find(database.images.set == 1 ) ;
val = find(database.images.set == 2 ) ;

% Setup data fetching options
bopts.numThreads = 1 ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,36,'single') ;
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = false;

% Launch SGD
[net,stats] = cnn_train_dag(net, database, getBatchWrapper(bopts), ...
                     train_opts, ....
                     'train', train, ...
                     'val', val, ...
                     opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,'prefetch',nargout==0) ;
