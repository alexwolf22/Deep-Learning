function [iscorrect, err] = mean_squared_error_test()
%MEAN_SQUARED_ERROR_TEST Unit tests for MEAN_SQUARED_ERROR.
%  [ISCORRECT, ERR] = MEAN_SQUARED_ERROR_TEST() provides unit tests for
%     MEAN_SQUARED_ERROR. There are several constants provided below:
%       TOL: error tolerance for backward direction, i.e. if the error is
%         equal to or higher than this value ISCORRECT is false.
%       DELTA: difference parameter used for computing finite difference.
%       X1: a 4-by-4-by-3-by-2 array of inputs
%       X3: a 4-by-4-by-3-by-2 array of inputs
%       DZDY: a 1-by-1 scalar of gradients of an arbitrary scalar Z with
%         respect to Y.
%
%     ISCORRECT is true if GENERALIZED_LOGISTIC passes all unit tests and is
%     false otherwise.
%
%     ERR is a struct object having the following fields:
%        ERR.dzdx1
%        ERR.dzdx2
%     The error between arbitrary arrays x and y is defined here as the maximum
%       value of the absolute differeence between x and y, i.e.
%       err(x,y) = max(abs(x(:)-y(:))).

%%%%%%%% DO NOT EDIT BELOW %%%%%%%%
S = load('mean_squared_error_test.mat');
X1 = S.X1;
X2 = S.X2;
DZDY = S.DZDY;
TOL = S.TOL;
DELTA = S.DELTA;
%%%%%%%% DO NOT EDIT ABOVE %%%%%%%%
[hInit, wInit, kInit, tInit]= size(X1);
iscorrect= true;

eLength= hInit*wInit*kInit*tInit;
er1=zeros(eLength,1);
er2=zeros(eLength,1);

eIndex=1;
%get analytical graient
[dzdx1, dzdx2]= mean_squared_error(X1, X2, DZDY);

for h=1:hInit
    for w=1:wInit
        for k=1:kInit
            for t=1:tInit

                A= zeros(hInit,wInit,kInit,tInit);
                A(h ,w, k, t)= DELTA;

                x1Plus=X1 +A;
                x1Minus=X1 -A;

                yx1p= mean_squared_error(x1Plus, X2);
                yx1m= mean_squared_error(x1Minus, X2);
               
                f1=((yx1p-yx1m)/(2*DELTA))*DZDY;
                ap1=dzdx1(h,w,k,t);
                
                error1= abs(ap1-f1);
                er1(eIndex)=error1;
                
                %comput x2 error
                
                x2Plus=X2 +A;
                x2Minus=X2 -A;
                
                yx2p= mean_squared_error(X1, x2Plus);
                yx2m= mean_squared_error(X1, x2Minus);
                
                f2=((yx2p-yx2m)/(2*DELTA))*DZDY;
                ap2=dzdx2(h,w,k,t);
                
                error2= abs(ap2-f2);       
                er2(eIndex)=error2;

                eIndex=eIndex+1;
            end
        end
    end
end

%check errors
if max(er1)>TOL
   fprintf('DXDX1 has incorrect error\n');
   iscorrect=false;
end

%check errors
if max(er2)>TOL
   fprintf('DXDX2 has incorrect error\n');
   iscorrect=false;
end

%init err stuct
err= struct('dzdx1', max(er1), 'dzdx2', max(er2));

%save results if correct
save('mean_squared_error_test_results.mat', 'iscorrect', 'err');


