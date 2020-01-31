function hess = hess_logistic(x, W, Y)


% Logistic Loss
n = length(Y);
hess = zeros(size(W,2));

for i=1:n
    w = W(i,:);
    y = Y(i);
    
    v = exp(-y* (w*x));
    hess = hess + (v) /(1 + v)^2 *((w')*w);
end