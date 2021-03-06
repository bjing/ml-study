function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

candidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
combos = zeros(length(candidates)^2, 3);

min_error = 100000;
for i = 1:length(candidates)
    for j = 1:length(candidates)
        % Values of C and sigma to try
        C = candidates(i);
        sigma = candidates(j);
        % Train model
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        % Make predicton
        predictions = svmPredict(model, Xval);
        % Calculate error
        error_cv = mean(double(predictions ~= yval))
        % Check whether C and sigma values are better
        if error_cv < min_error
            min_error = error_cv;
            best_c = C;
            best_sigma = sigma;
        endif
    endfor
endfor


C = best_c;
sigma = best_sigma;



% =========================================================================

end
