function varargout = fully_connected(x, w, b, dzdy)
%FULLY_CONNECTED Provides a fully connected layer.
%  Y = FULLY_CONNECTED(X, W, B) returns an affine transformation of X
%    using parameters W and B. X is a H-by-W-by-K-by-T array where the first
%    three dimensions represent height, width, and channels respectively; the
%    last dimension represents batch size. W is an M-by-H-by-W-by-K weight array
%    where the last three dimenions represent height, width, and channels
%    respectively; the first dimension represents number of projection
%    outputs. Y is returned as a 1-by-1-by-M-by-T array. B is a M-by-1
%    column vector of biases.
%  [DZDX, DZDW, DZDB] = FULLY_CONNECTED(X, W, B, DZDY)
%     Backpropagates the gradient DZDY with respect to X, W, and B.

%cast everything from double to single
[h, width, k, t]=size(x);
[m, ~, ~, ~]=size(w);
n= h*width*k;

newX= reshape(x,[n, t]);
newW= reshape(w, [] ,n);

switch nargin
    %return affine transformation
    case 3
        wx=(newW* newX);
        aOfx=bsxfun(@plus, wx, b);
        Y= ones(1,1,m,t);
        Y(1,1,:,:)= aOfx;
        varargout={Y};
        
    %backpropagate gradient with respect to inputs
    case 4
        dzdy=single(dzdy); %this function needed to be commented out so fully connected worked with
                            %matconvnet.  See fully_connected2()
        newdY= reshape(dzdy,[],t);
        
        DZDW= newdY*newX';
        DZDX= newdY'*newW;
        DZDB= {reshape(sum((dzdy), 4),[m,1])};
        DZDW= {reshape(DZDW, [m, h, width,k])};
        DZDX= {reshape(DZDX', [h,width,k, t])};
        
        varargout=[DZDX, DZDW, DZDB];
        
end
