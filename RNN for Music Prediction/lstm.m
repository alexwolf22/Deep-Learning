function varargout = lstm(f, i, o, a, c, dzdy, dzdc)
%LSTM Provides a long short term memory actvation.
%  Y,CNEW = LSTM(I, F, O, A, C) returns the gated output Y and new cell
%     state CNEW for a LSTM activation function. I, F, and O are the
%     pre-activations for the forget, input, and ouput gates. A are the
%     pre-actvation weights. C is the initial cell state. Hence
%     CNEW = TANH(A).*SIGMA(I) + SIGMA(F).*C and Y = SIGMA(O) .*
%     TANH(CNEW), where SIGMA is the sigmoid function and TANH is the hyperbolic
%     tangent function.
%  [DZDI, DZDF, DZDO, DZDA, DZDC] = FULLY_CONNECTED(I, F, O, A, C, DZDY, DZDC)
%     Backpropagates the gradients DZDY and DZDC with respect to I, F, O,
%     A, and C.

  % compute the input gate
  ikxt= generalized_logistic(i, 0, 1, 1);

  % compute the forget gate
  fkxt= generalized_logistic(f, 0, 1, 1);

  % compute the output gate
  okxt= generalized_logistic(o, 0, 1, 1);

  % compute the activation
  akxt= generalized_logistic(a, -1, 1, 2);

  % compute the new cell state using the gates and activation
  c1= ikxt.* akxt;
  c2= fkxt.* c;
  cKur= c1+ c2;

  % compute the layer layer output
  y= okxt.* generalized_logistic(cKur, -1, 1, 2);
  
  if nargin < 6
    % return the layer output and cell for forward direction
    varargout = {y, cKur};
  else
    % compute and return the gradients with respct to each of the input
    % arguments
    
    toAdd= dzdc;
    dzdc= okxt.*generalized_logistic(cKur, -1, 1, 2, dzdy);
    dzdc= dzdc+toAdd;
    
    dzdf= dzdc.*c;
    dzdi= dzdc.* akxt;
    dzdo= dzdy.* generalized_logistic(cKur, -1, 1, 2);
    dzda= dzdc.* ikxt;
    
    dzdf= generalized_logistic(f, 0, 1, 1, dzdf);
    dzdi= generalized_logistic(i, 0, 1, 1, dzdi);
    dzdo= generalized_logistic(o, 0, 1, 1, dzdo);
    dzda= generalized_logistic(a, -1, 1, 2, dzda);
    
    varargout = {dzdi, dzdf, dzdo, dzda, dzdc};
  end

end
