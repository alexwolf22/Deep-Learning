classdef GeneralizedLogistic < dagnn.ElementWise
  properties
  % 1-D array of input settings.
  % Argument 1 is l
  % Argument 2 is u
  % Argument 3 is g
  % default is tanh
    l = -1;
    u = 1;
    g = 2;
  end
  methods
  
    function outputs = forward(obj, inputs, ~)
      outputs{1} = generalized_logistic(inputs{1}, obj.l, obj.u, obj.g) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
      derInputs{1} = generalized_logistic(inputs{1}, obj.l, obj.u, obj.g, ...
        derOutputs{1}) ;
      derParams = {} ;
    end
    
    function obj = GeneralizedLogistic(varargin)
      obj.load(varargin) ;
      obj.l = obj.l ;
      obj.u = obj.u ;
      obj.g = obj.g ;  
    end
  end
end
