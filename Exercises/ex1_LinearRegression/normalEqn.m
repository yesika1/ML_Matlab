
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%
% ============================================================


function [theta] = normalEqn(X, y)

theta = zeros(size(X, 2), 1);
theta = pinv(X'*X) * X'*y;

end
