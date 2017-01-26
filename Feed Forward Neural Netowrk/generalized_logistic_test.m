function [iscorrect, err] = generalized_logistic_test()
%GENERALIZED_LOGISTIC_TEST Unit tests for GENERALIZED_LOGISTIC.
%  [ISCORRECT, ERR] = GENERALIZED_LOGISTIC_TEST() provides unit tests for
%     GENERALIZED_LOGISTIC. There are several constants provided below:
%       TOL1: error tolerance for forward direction, i.e. if the error is
%         equal to or higher than this value ISCORRECT is false.
%       TOL2: error tolerance for backward direction.
%       DELTA: difference parameter used for computing finite difference.
%       X: a 4-by-4-by-3-by-2 array of inputs.
%       L, U, and G: the parameter values necessary to compute the
%         hyperbolic tangent using GENERALIZED_LOGISTIC.
%       DZDY: a 4-by-4-by-3-by-2 array of gradientsof an arbitrary scalar Z with
%         respect to Y.
%
%     ISCORRECT is true if GENERALIZED_LOGISTIC passes all unit tests and is
%     false otherwise.
%
%     ERR is a struct object having the following fields:
%        y
%        dzdx
%        dzdl
%        dzdu
%        dzdg
%     The error between arbitrary arrays x and y is defined here as the maximum
%       value of the absolute differeence between x and y, i.e.
%       err(x,y) = max(abs(x(:)-y(:))).

%%%%%%%% DO NOT EDIT BELOW %%%%%%%%
S = load('generalized_logistic_test.mat');

TOL1  = S.TOL1;
TOL2  = S.TOL2;
DELTA = S.DELTA;

X = S.X;

% parameter settings for the hyperbolic tangent
L = S.L;
U = S.U;
G = S.G;

% three sets of gradients, each corresponding to a set of input
DZDY = S.DZDY;
%%%%%%%% DO NOT EDIT ABOVE %%%%%%%%
[hInit, wInit, kInit, tInit]= size(X);

exh= hInit*wInit*kInit*tInit;

eNx=ones(exh,1);
eAx=ones(exh,1);

[dzdxA, dzdlA, dzduA, dzdgA]=generalized_logistic(X , L, U, G, DZDY);

%compute all p's of x
eXIndex=1;
for h=1:hInit
    for w=1:wInit
        for k=1:kInit
            for t=1:tInit
                
                %get analytical gradient
                dA=dzdxA(h, w, k, t);
                eAx(eXIndex)=dA;
                
                A= zeros(hInit,wInit,kInit,tInit);
                A(h ,w, k, t)= DELTA;

                xPlus=X +A;
                xMinus=X -A;

                %get n gradient
                yP= generalized_logistic(xPlus, L, U, G);
                yM= generalized_logistic(xMinus, L, U, G);
                f= (yP-yM)/(2*DELTA);
                dnX= sum(sum(sum(sum(DZDY.*f))));
                eNx(eXIndex)=dnX;
                eXIndex=eXIndex+1;
                
            end
        end
    end
end

%compute p's of u, g, and l
lPlus=L +DELTA;
lMinus=L -DELTA;

uPlus=U +DELTA;
uMinus=U -DELTA;

gPlus=G +DELTA;
gMinus=G -DELTA;

yP= generalized_logistic(X, lPlus, U, G);
yM= generalized_logistic(X, lMinus, U, G);
f= (yP-yM)/(2*DELTA);
dnL= sum(sum(sum(sum(DZDY.*f))));

yP= generalized_logistic(X, L, uPlus, G);
yM= generalized_logistic(X, L, uMinus, G);
f= (yP-yM)/(2*DELTA);
dnU= sum(sum(sum(sum(DZDY.*f))));

yP= generalized_logistic(X, L, U, gPlus);
yM= generalized_logistic(X, L, U, gMinus);
f= (yP-yM)/(2*DELTA);
dnG= sum(sum(sum(sum(DZDY.*f))));

%compute y
y=generalized_logistic(X, L, U, G);
yTan= tanh(X);

%init return values
yError= abs(yTan-y);
xError= abs(eAx-eNx);
lError= abs(dzdlA-dnL);
uError= abs(dzduA-dnU);
gError= abs(dzdgA-dnG);

%init return values
iscorrect=true;

%check if generalized_logistic was right
yEr=max(yError);
if max(yEr)>TOL1
    fprintf('There is an error in y\n');
    iscorrect=false;
end

xEr=max(xError);
fprintf('%d-is max dzdx\n',xEr);
if max(xEr)>TOL2
    fprintf('There is an error in dzdx\n');
    iscorrect=false;
end

if lError>TOL2
    fprintf('There is an error in dzdl\n');
    iscorrect=false;
end

if uError>TOL2
    fprintf('There is an error in dzdu\n');
    iscorrect=false;
end

if gError>TOL2
    fprintf('There is an error in dzdg\n');
    iscorrect=false;
end

err= struct('y', y, 'dzdx', xEr, 'dzdl', lError, 'dzdu', uError, 'dzdg', gError);

save('generalized_logistic_test_results.mat', 'iscorrect', 'err');
