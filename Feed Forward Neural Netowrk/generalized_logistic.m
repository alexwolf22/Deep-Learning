function varargout = generalized_logistic(x, l, u, g, dzdy)
%GENERALIZED_LOGISTIC Generalized logistic function.
%  Y = GENERALIZED_LOGISTIC(X, L, U, G) computes the generalized logistic
%    function. X is a 4-D array where the first three dimensions represent
%    height, width, and channels respectively; the last dimension represents
%    batch size. L and U are scalars representing the asymptotic lower and
%    upper bounds of the logistic function. G is a scalar representing the
%    growth rate of the logistic function. Y is a 4-D array having the same
%    size as X.
%
%  [DZDX, DZDL, DZDU, DZDG] = GENERALIZED_LOGISTIC(X, L, U, G, DZDY)
%     Backpropagates the gradient DZDY with respect to X, L, U, and G.
%
%  Setting for the logistic sigmoid function:
%    L = 0;
%    U = 1;
%    G = 1;
%
%  Setting for the hyperbolic tangent function:
%    L = -1;
%    U = 1;
%    G = 2;

switch nargin
    %compute y
    case 4
        
        Y= l+((u-l)/(1+exp(-g*x)));
        varargout={Y};
    case 5
        
        DZDY= reshape(dzdy, 1, []);
        
        DYDX= ((((u-l)*g)*(exp(g*x))))./((1+exp(g*x)).^2);
        DYDL= reshape((1-(1/((1+exp(-g*x))))), [], 1);
        DYDG= reshape((((u-l)*x).*exp(g*x))./((1+exp(g*x)).^2),[], 1);
        DYDU= reshape(1/(1+exp(-g*x)),[], 1);
        
        DZDX= dzdy.*DYDX;
        DZDL= DZDY*DYDL;
        DZDG= DZDY*DYDG;
        DZDU= DZDY*DYDU;
        
        varargout=[{DZDX}, {DZDL}, {DZDU}, {DZDG}];
end
