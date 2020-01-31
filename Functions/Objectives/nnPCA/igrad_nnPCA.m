function grad = igrad_nnPCA(x, i, W)
    
    m = size(W,1);
    
    grad = -W(i,:)'*(W(i,:)*x)/m;

end