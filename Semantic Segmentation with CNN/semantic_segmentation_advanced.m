function dagnn_net = semantic_segmentation_advanced(netspec_opts, train_opts)


dagnn_net= dagnn.DagNN();
dagnn_net.meta.trainOpts = train_opts;
dagnn_net.meta.inputSize= [32, 32, 3];

% Set up the model based on netspec_opts
kSize= netspec_opts.kernel_size;
numFilts= netspec_opts.num_filters;
strides= netspec_opts.stride;
layerTypes= netspec_opts.layer_type;
input= netspec_opts.input;

[~, L] = size(strides);
prevName= 'input';
index4name=1;
lastConvFilt=0;
%loop through all layers
for i=1:L
    
    type= layerTypes{i};
    if strcmp(type, 'convt')
        layer_name= strcat(type, '_', num2str(kSize(1, i)), 'x_', num2str(index4name));
    else
        layer_name= strcat(type, '_', num2str(index4name));
    end
    
    %add conv layer
    if strcmp(type, 'conv')
        ksizeW= kSize(1, i);
        ksizeH= kSize(2, i);
        numfilt= numFilts(i);
        stride= strides(i);
        pad= (ksizeW-1)/2;
        
        wParam= strcat(layer_name, '_w');
        bParam= strcat(layer_name, '_b');
        
        lastConvFilt=3;
        if i~=1
            lastConvFilt= numFilts(i-3);
        end
        
        params= {wParam, bParam};
        
        dagnn_net.addLayer(layer_name, ...
            dagnn.Conv('size', [ksizeW, ksizeH, lastConvFilt, numfilt],...
            'stride', stride,...
            'pad', pad,...
            'hasBias', true),...
            prevName,...
            layer_name,...
            params);
        
        lastConvFilt= numfilt;
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
    
    %add convT layer
    if strcmp(type, 'convt')
        
        %get filters 
        numChannels= numFilts(i);
        numGroups= numChannels;
        upsample= kSize(1, i);
        
        filters= single( bilinear_u(upsample, numGroups, numChannels));
        params_name= {strcat(layer_name, '_f')};
        
        %if this is last upsample make output called pred
        if i==35
            outputName= 'pred';
        else
            outputName=layer_name;
        end
        
        dagnn_net.addLayer(layer_name,...
            dagnn.ConvTranspose(...
                'size', size(filters),...
                'upsample', upsample,...
                'crop', [0,0,0,0],...
                'numGroups', lastConvFilt,...
                'hasBias', false),...
                prevName, outputName, params_name);
        
        %initial filter weights
        paramIndex= dagnn_net.getParamIndex(params_name);
        dagnn_net.params(paramIndex).value = filters;
        dagnn_net.params(paramIndex).learningRate = 0.1;
        dagnn_net.params(paramIndex).weightDecay = 1 ;
        
        index4name=index4name+1;
    end
    
    %add skip layer %in adv model both have layer input of 64
    if strcmp(type, 'skip')
        
        wParam= strcat(layer_name, '_w');
        bParam= strcat(layer_name, '_b');
        params_name= {wParam, bParam};
       
        prevLayerFilt=64;

        preLayer=input(1,i);
        
        dagnn_net.addLayer(layer_name, ...
            dagnn.Conv('size', [1, 1, prevLayerFilt, lastConvFilt],...
            'stride', 1,...
            'pad', 0,...
            'hasBias', 1),...
            preLayer,... 
            layer_name,...
            params_name);
        
        %pre-init weights and bias
        wIndex= dagnn_net.getParamIndex(params_name{1});
        dagnn_net.params(wIndex).value= .1* rand(1, 1, prevLayerFilt, lastConvFilt, 'single');
        dagnn_net.params(wIndex).learningRate = 0.1;
        dagnn_net.params(wIndex).weightDecay = 1 ;
        
        bIndex= dagnn_net.getParamIndex(params_name{2});
        dagnn_net.params(bIndex).value= zeros(1, 1, lastConvFilt, 'single');
        dagnn_net.params(bIndex).learningRate = 2;
        dagnn_net.params(bIndex).weightDecay = 1 ;
    end
    
    %add sum layer
    if strcmp(type, 'sum')
        fisrtIn= input(1,i);
        secondIn= input(2,i);
        inputss={fisrtIn{1}, secondIn{1}};
        
        dagnn_net.addLayer(layer_name,...
            dagnn.Sum(),...
            inputss,...
            layer_name);
    end
    
    prevName= layer_name;
    
    %init params after first 13 layers
    if i==28
        dagnn_net.initParams();
    end
    
end


% Initialize the parameters of all the layers so far
% Set training policy


% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------
dagnn_net.addLayer('objective', SegmentationLoss('loss', 'softmaxlog'),...
    {'pred', 'label'}, 'objective');

dagnn_net.addLayer('accuracy', SegmentationAccuracy(),...
    {'pred', 'label'}, 'accuracy');
                   
% Make the output of the bilinear interpolator is not discared for
% visualization purposes
dagnn_net.vars(dagnn_net.getVarIndex('pred')).precious = 1 ;


