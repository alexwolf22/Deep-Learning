function varargout = mean_squared_error(x1, x2, dzdy)
%MEAN_SQUARED_ERROR Computes the MSE loss between two arrays.
%  Y = FULLY_CONNECTED(X1, X2) returns the MSE loss between X1 and X2. X1
%    and X2 are each 4-D arrays where the first three dimensions represent
%    height, width, and channels respectively; the last dimension represents
%     batch size.
%  [DZDX1, DZDX2] = MEAN_SQUARED_ERROR(X1, X2, DZDY)
%     Backpropagates the gradient DZDY with respect to X1, and X2.
[h, w, k, t]= size(x1);
m=h*w*k*t;

switch nargin
    %return mean squared error
    case 2
        varargout={sum(sum(sum(sum((x1-x2).^(2)))))/m};
        
    %return gradients
    case 3
        DZDX1= dzdy.*(2*(x1-x2)/m);
        DZDX2= dzdy.*(-2*(x1-x2)/m);
        varargout= [{DZDX1}, {DZDX2}];
end


