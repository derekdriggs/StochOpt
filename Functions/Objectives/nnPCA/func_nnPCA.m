function vfun = func_nnPCA(x, W, Y, lambda)
    
    m    = size(W,1);
    vfun = -1/(2*m)*norm(W*x - Y)^2 - lambda/2*norm(x(:))^2;

end