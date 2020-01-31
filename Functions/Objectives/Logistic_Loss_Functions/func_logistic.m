function vfun = func_logistic(x, W, Y)


% Logistic Loss
n = length(Y);
vfun = 0;

for i=1:n
    w = W(i,:);
    y = Y(i);
    
    vfun = vfun + log( 1 + exp(-y* (w*x)) );
end

vfun = vfun /n;