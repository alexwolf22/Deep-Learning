function dagnn_net = cnn_categorization_advanced(netspec_opts, train_opts, class_names)

% Create an instance of the dagnn class
dagnn_net= dagnn.DagNN();

% Specify the input size
dagnn_net.meta.inputSize= [30, 30, 3];

% Set up the model based on netspec_opts
kSize= netspec_opts.kernel_size;
numFilts= netspec_opts.num_filters;
strides= netspec_opts.stride;
layerTypes= netspec_opts.layer_type;

[~, L] = size(strides);
prevName= 'input';
index4name=1;

%loop through all layers
for i=1:L
    
    type= layerTypes{i};
    layer_name= strcat(type, '_', num2str(index4name));
    
    %add conv layer
    if strcmp(type, 'conv')
        
        ksizeW= kSize(1, i);
        ksizeH= kSize(2, i);
        numfilt= numFilts(i);
        stride= strides(i);
        pad= (ksizeW-1)/2;
        
        wParam= strcat(layer_name, '_w');
        bParam= strcat(layer_name, '_b');
        
        prevLayerFilt=3;
        if i~=1
            prevLayerFilt= numFilts(i-3);
        end
        
        params= {wParam, bParam};
        
        dagnn_net.addLayer(layer_name, ...
            dagnn.Conv('size', [ksizeW, ksizeH, prevLayerFilt, numfilt],...
            'stride', stride,...
            'pad', pad,...
            'hasBias', true),...
            prevName,...
            layer_name,...
            params);
    end
    
    %add batch norm layer
    if strcmp(type, 'bn')
        wParam= strcat(layer_name, '_w');
        bParam= strcat(layer_name, '_b');
        mParam= strcat(layer_name, '_m');
        params= {wParam, bParam, mParam};
        
        numChan=numFilts(i);
        
        dagnn_net.addLayer(layer_name, ...
            dagnn.BatchNorm('numChannels', numChan, 'epsilon', .00001),...
            prevName,...
            layer_name,...
            params);
    end
    
    %add Rectified Linear Unit Layer
    if strcmp(type, 'relu')
        index4name=index4name+1;
        
        dagnn_net.addLayer(layer_name, ...
            dagnn.ReLU(),...
            prevName,...
            layer_name);
    end
    
    %add pooling layer
    if strcmp(type, 'pool')
        
        psizeW= kSize(1, i);
        psizeH= kSize(2, i);
        psize= [psizeW, psizeH];
        
        dagnn_net.addLayer(layer_name, ...
        dagnn.Pooling('poolSize', psize, 'method', 'avg'),...
        prevName,...
        layer_name);
        
    end
    
    prevName= layer_name;
   
end

% Add a pooling layer that performs prediction of the correct size (1x1 spatial size)


% Add a final convolutional layer that produces the outputs for each example with K channels
ksizeW= 1;
ksizeH= 1;
numfilt= 16;
stride= 1;
pad= 0;
layer_name= 'pred';

wParam= strcat(layer_name, '_w');
bParam= strcat(layer_name, '_b');
params= {wParam, bParam};

prevLayerFilt=0;
if i~=1
    prevLayerFilt= 256;
end

dagnn_net.addLayer(layer_name, ...
            dagnn.Conv('size', [ksizeW, ksizeH, prevLayerFilt, numfilt],...
            'stride', stride,...
            'pad', pad,...
            'hasBias', true),...
            prevName,...
            layer_name,...
            params); 

layerParamNamesLoss= cell(1,2);
layerParamNamesLoss(1)= {layer_name};
layerParamNamesLoss(2)= {'label'};
obj= 'objective';
top1='top1error';

dagnn_net.addLayer(obj, dagnn.Loss('loss', 'softmaxlog'), layerParamNamesLoss, obj, {});
dagnn_net.addLayer(top1, dagnn.Loss('loss', 'classerror'), layerParamNamesLoss, top1, {});

% Set training policy
dagnn_net.meta.classes.names= class_names;
dagnn_net.meta.trainOpts = train_opts;

% Initialize the parameters of your model
dagnn_net.initParams();
    
end

