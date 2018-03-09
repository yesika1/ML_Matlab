
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% =========================================================================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% The hypothesis (also called the prediction) is simply the product of X and theta.
% Since X is size (m x n) and theta is size (n x 1), you arrange the order of operators so the result is size (m x 1).
% =========================================================================


function J = computeCost(X, y, theta)

m = length(y); 		% number of training examples
J = 0;

h = X * theta;              % predictions of hypothesis on examples
sqrErrors = (h - y).^2; % squared errors

J = 1/(2*m) * sum(sqrErrors);


end
