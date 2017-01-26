function dagnn_net = create_dagnn_net(netspec_opts, train_opts)
% This function constructs a dagnn class object based on your specifications.
% Inputs:
% 1- netspec_opts: a struct which contains information used for constructing the
% net.
% 2- train_opts: a struct which contains information used for training
% your proposed model.
% Outputs:
% 1- dagnn_net: a class object which contains the network architecture
% you specified.

% Create an instance of the dagnn class
dagnn_net= dagnn.DagNN();

% Set up the model based on netspec_opts
nonLin= {netspec_opts.non_linearity};
hiddenUnits= netspec_opts.hidden_units;
numFeats= netspec_opts.input_features;

%get fc params name 1x2 cell 
layerParamNamesFC= cell(1,2);

%get empty param names cell for GL
layerParamNamesGL= {};

%for each layer
[~, L]= size(nonLin);
prevLayerName='input';
for i=1:L
    
    %make sure all layers are correctly inputed
    if strcmp(nonLin(i), 'tanH') && strcmp(nonLin(i), 'sigmoid')
        fprintf('the layer name %s, at hidden level %d, was not reconginzed\n',nonLin{i}, i);
        exit
    end
    
   %get input size vector for each fully_connected in each layer
   inputSize=ones(1,4);
   inputSize(1)= hiddenUnits(i);
   if i==1
       inputSize(4)=numFeats;
   else
       inputSize(4)= hiddenUnits(i-1);
   end
   layer_Name= strcat('fc_',num2str(i)); 
   layerParamNamesFC(1,1)= {strcat(layer_Name,'_W')};
   layerParamNamesFC(1,2)= {strcat(layer_Name,'_b')};
   
   dagnn_net.addLayer(layer_Name, FullyConnected('size', inputSize), prevLayerName, layer_Name, layerParamNamesFC);
   prevLayerName=layer_Name;
   
    %Now add GL layer
    %get specific layer data
    if strcmp(nonLin{i}, 'sigmoid')
        layer_Name= strcat('sigmoid_',num2str(i));
        l=0;
        u=1;
        g=1;
    elseif strcmp(nonLin{i}, 'tanH')
        layer_Name= strcat('tanH_', int2str(i));
        l=-1;
        u=1;
        g=2;
    end
    
    dagnn_net.addLayer(layer_Name, GeneralizedLogistic('l', l, 'u', u, 'g', g), prevLayerName, layer_Name, layerParamNamesGL);
    prevLayerName=layer_Name;
end

% Add a final layer that performs prediction of the correct size
finalOutSize= netspec_opts.final_output_size;
inputSize=ones(1,4);
inputSize(1)=finalOutSize;
inputSize(4)= hiddenUnits(L);

layer_Name= 'prediction';

layerParamNamesPR=cell(1,2);
layerParamNamesPR(1,1)= {'pred_b'};
layerParamNamesPR(1,2)= {'pred_w'};

dagnn_net.addLayer(layer_Name, FullyConnected('size', inputSize), prevLayerName, layer_Name, layerParamNamesPR);

% Add layers
lossFx= netspec_opts.loss_function;

layerParamNamesLoss= cell(1,2);
layerParamNamesLoss(1)= {layer_Name};
layerParamNamesLoss(2)= {'label'};
obj= 'objective';
top1='top1error';

dagnn_net.addLayer(obj, dagnn.Loss('loss', lossFx), layerParamNamesLoss, obj, layerParamNamesGL);
dagnn_net.addLayer(top1, dagnn.Loss('loss', 'classerror'), layerParamNamesLoss, top1, layerParamNamesGL);

% Don't forget to initialize the parameters of your model
dagnn_net.meta.trainOpts = train_opts;
dagnn_net.initParams();

