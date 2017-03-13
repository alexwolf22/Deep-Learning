function [iscorrect, err] = lstm_test()
%%%%% Set RNG for repeatability
rng(1234);
%%%%%

I = randn(1, 1, 20,32);
F = randn(1, 1, 20,32);
O = randn(1, 1, 20,32);
A = randn(1, 1, 20,32);
C = randn(1, 1, 20,32);
DELTA = 1e-06;
TOL = 1e-8;

iscorrect = true;

y = lstm(I, F, O, A, C);

dzdi_n = zeros(numel(y), numel(I));
dzdf_n = zeros(numel(y), numel(F));
dzdo_n = zeros(numel(y), numel(O));
dzda_n = zeros(numel(y), numel(A));
dzdc_n = zeros(numel(y), numel(C));

for i = 1:numel(I)
    i_test0 = I;
    i_test1 = I;
    i_test0(i) = i_test0(i) + DELTA;
    i_test1(i) = i_test1(i) - DELTA;
    dzdi0 = lstm(i_test0, F, O, A, C);
    dzdi1 = lstm(i_test1, F, O, A, C);
    dzdi_n(:, i) =  (dzdi0(:) -  dzdi1(:)) ./ (2*DELTA);
end
 
for i = 1:numel(F)
    f_test0 = F;
    f_test1 = F;
    f_test0(i) = f_test0(i) + DELTA;
    f_test1(i) = f_test1(i) - DELTA;
    dzdf0 = lstm(I, f_test0, O, A, C);
    dzdf1 = lstm(I, f_test1, O, A, C);
    dzdf_n(:, i) =  (dzdf0(:) -  dzdf1(:)) ./ (2*DELTA);
end

for i = 1:numel(O)
    o_test0 = O;
    o_test1 = O;
    o_test0(i) = o_test0(i) + DELTA;
    o_test1(i) = o_test1(i) - DELTA;
    dzdo0 = lstm(I, F, o_test0, A, C);
    dzdo1 = lstm(I, F, o_test1, A, C);
    dzdo_n(:, i) =  (dzdo0(:) -  dzdo1(:)) ./ (2*DELTA);
end

for i = 1:numel(A)
    a_test0 = A;
    a_test1 = A;
    a_test0(i) = a_test0(i) + DELTA;
    a_test1(i) = a_test1(i) - DELTA;
    dzda0 = lstm(I, F, O, a_test0, C);
    dzda1 = lstm(I, F, O, a_test1, C);
    dzda_n(:, i) =  (dzda0(:) -  dzda1(:)) ./ (2*DELTA);
end

for i = 1:numel(C)
    c_test0 = C;
    c_test1 = C;
    c_test0(i) = c_test0(i) + DELTA;
    c_test1(i) = c_test1(i) - DELTA;
    dzdc0 = lstm(I, F, O, A, c_test0);
    dzdc1 = lstm(I, F, O, A, c_test1);
    dzdc_n(:, i) =  (dzdc0(:) -  dzdc1(:)) ./ (2*DELTA);
end

dzdi_n = reshape(sum(dzdi_n, 1), size(I));
dzdf_n = reshape(sum(dzdf_n, 1), size(F));
dzdo_n = reshape(sum(dzdo_n, 1), size(O));
dzda_n = reshape(sum(dzda_n, 1), size(A));
dzdc_n = reshape(sum(dzdc_n, 1), size(A));
[dzdi_t, dzdf_t, dzdo_t, dzda_t, dzdc_t] = lstm(I, F, O, A, C, 1, 0);

[c, e] = test(dzdi_n, dzdi_t, TOL);
iscorrect = iscorrect && c;
err.dzdi = e;

[c, e] = test(dzdf_n, dzdf_t, TOL);
iscorrect = iscorrect && c;
err.dzdf = e;

[c, e] = test(dzdo_n, dzdo_t, TOL);
iscorrect = iscorrect && c;
err.dzdo = e;

[c, e] = test(dzda_n, dzda_t, TOL);
iscorrect = iscorrect && c;
err.dzda = e;

[c, e] = test(dzdc_n, dzdc_t, TOL);
iscorrect = iscorrect && c;
err.dzdc = e;
end

function [isclose, err] = test(x, y, tol)
  err = max(abs(x(:) - y(:)));
  isclose = err < tol;
end