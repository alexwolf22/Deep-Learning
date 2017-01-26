classdef FullyConnected < dagnn.Filter
  properties
    size = [0 0 0 0]
  end

  methods
    function outputs = forward(~, inputs, params)
        outputs{1} = fully_connected(inputs{1}, params{1}, params{2});    
    end

    function [derInputs, derParams] = backward(~, inputs, params, derOutputs)
      [derInputs{1}, derParams{1}, derParams{2}] = ...
        fully_connected(inputs{1}, params{1}, params{2}, derOutputs{1});    
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
      % Xavier improved
        sc = sqrt(2 / prod(obj.size(1:3))) ;
        params{1} = randn(obj.size,'single') * sc ;
        params{2} = zeros(obj.size(1),1,'single') ;
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end
    
    function obj = FullyConnected(varargin)
      obj.load(varargin) ;
      obj.size = obj.size ;
    end

  end
  
end
