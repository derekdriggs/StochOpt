function vfun = func_lasso(x, W, Y, lambda)
    
    m    = size(W,1);
    vfun = 1/(2*m)*norm(W*x - Y)^2 + lambda*norm(x,1);

end