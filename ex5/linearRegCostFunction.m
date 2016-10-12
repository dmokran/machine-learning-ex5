function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X = 12x2
% y = 12x1
% theta = 2x1
oneDiv2m = 1 / (2*m);
lDiv2m = lambda / (2*m);
lDivm = lambda / m;
errTerm = sum((X*theta - y).^2); %12x2 * 2x1 => 12x1 - 12x1 = 12x1
t1 = theta(1);
t2end = theta(2:end);
regTerm = sum([t2end.^2]);
J = oneDiv2m * errTerm + lDiv2m * regTerm;
gradTerm = (1/m) * (X' * (X*theta - y)); % 2x12 * (12x1) = 2x1
gradRegTerm = lDivm * t2end;
grad = [gradTerm(1); gradTerm(2:end) + gradRegTerm];

% =========================================================================

grad = grad(:);

end
