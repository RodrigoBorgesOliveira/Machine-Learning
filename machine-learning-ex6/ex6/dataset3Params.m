function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
cIndex = 0.01;
sigmaIndex = 0.01;

% Get the initial value for the minimum error (using values C and sigma = 0.01).
model = svmTrain(X, y, cIndex, @(x1, x2) gaussianKernel(x1, x2, sigmaIndex));
predictions = svmPredict(model, Xval);
     
minimumError = mean(double(predictions ~= yval));

steps = [0.03 0.1 0.3 1 3 10 30];
cStepsIndex = 1;

cIndex = steps(cStepsIndex); % Starting value of cIndex on the loop.

while cIndex < 30
    sigmaStepsIndex = 1;
    sigmaIndex = steps(sigmaStepsIndex); % Starting value of sigmaIndex on the loop.
    while sigmaIndex < 30
        model = svmTrain(X, y, cIndex, @(x1, x2) gaussianKernel(x1, x2, sigmaIndex));
        predictions = svmPredict(model, Xval);
        
        predictionsError = mean(double(predictions ~= yval));
        
        % Verify if the new predictions error found is the minimum.
        if predictionsError < minimumError
            % It is, update the minimum error, C and sigma values.
            minimumError = predictionsError;
            C = cIndex;
            sigma = sigmaIndex;
        end;
        
        sigmaStepsIndex = sigmaStepsIndex + 1; % Increment sigma steps to get
        % next value of the steps.
        sigmaIndex = steps(sigmaStepsIndex); % Change sigmaIndex value.
    end;
    
    cStepsIndex = cStepsIndex + 1; % Increment c steps to get next value of the steps.
    cIndex = steps(cStepsIndex); % Change cIndex value.
end;


% =========================================================================

end
