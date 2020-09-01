function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

g = sigmoid(theta'*X');	% 1 x m vector

J = ((-1) *( (1/m) * sum((log(g) * y ) + (log(1-g) * (1-y)) ))) + sum((lambda/(2*m)) * ((theta(2 : n)).^2));	


for j = 1

grad(j) = (1/m) * ((g' - y)' * X(:,1));

end


for j = 2:n

grad(j) = ((1/m) * ((g' - y)' * X(:,j))) + ((lambda/m)*theta(j));

end




% =============================================================

end
