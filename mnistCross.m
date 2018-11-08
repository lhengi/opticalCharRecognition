%% Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance
%% This file is modified from pmtk3.googlecode.com

load('mnistData');

% set training & testing 

trainndx = 1:10000; 

folds = [3 10 50 100 1000];
errs = zeros(1,length(folds));
for i = 1:length(folds)
   
    errs(i) = myFunction(folds(i),1000,mnist);
    fprintf("Fold: %i  Error Rate: %.2f\n",folds(i),errs(i));
    
end





%display(length(folds));
%display(errs);
plot(folds,errs);
xlabel('fold');
ylabel('errors');

% function to setup function that accepts different fold number and do
% cross validation
function errorRates = myFunction(fold,trainLen,mnist)
  
  intervalSize = floor(trainLen/fold);
  testLowIndex = 1;
  testHighIndex = intervalSize;
  errorRates = 0.0;
  ntrain = trainLen - intervalSize;
  ntest = intervalSize;
  for i = 1:fold
      
      % do cross validation by manipulating train_image indices
      % manually handle edge case, when low index =1 and low index = 1000
      if (testLowIndex == 1)
          Xtrain = double(reshape(mnist.train_images(:,:,testHighIndex+1:trainLen),28*28,ntrain)');
          ytrain = (mnist.train_labels(testHighIndex+1:trainLen));
      elseif (testLowIndex == 1000)
          Xtrain = double(reshape(mnist.train_images(:,:,1:testLowIndex-1),28*28,ntrain)');
          ytrain = (mnist.train_labels(1:testLowIndex-1));
      else
        Xtrain = double(reshape(mnist.train_images(:,:,[1:testLowIndex-1 testHighIndex+1:trainLen]),28*28,ntrain)');
        ytrain = (mnist.train_labels([1:testLowIndex-1 testHighIndex+1:trainLen]));
      end
      
      Xtest  = double(reshape(mnist.test_images(:,:,testLowIndex:testHighIndex),28*28,ntest)');
      ytest  = (mnist.test_labels(testLowIndex:testHighIndex));
      
      

      % Precompute sum of squares term for speed
      XtrainSOS = sum(Xtrain.^2,2);
      XtestSOS  = sum(Xtest.^2,2);

      % fully solution takes too much memory so we will classify in batches
      % nbatches must be an even divisor of ntest, increase if you run out of memory 
      %if ntest > 1000
      %  nbatches = 50;
      %else
      %  nbatches = 5;
      %end
      %batches = mat2cell(1: idivide(ntest,1),1,(ntest/nbatches)*ones(1,nbatches));
      ypred = zeros(ntest,1);

      % Classify
      %for i=1:nbatches    
      dst = sqDistance(Xtest,Xtrain,XtestSOS,XtrainSOS);
      [junk,closest] = min(dst,[],2);
      ypred = ytrain(closest);
      %end
      % Report

      errorRate = mean(ypred ~= ytest);
      errorRates = errorRates + errorRate;
      %fprintf('Error Rate: %.2f%%\n',100*errorRate);
      
      testLowIndex = testHighIndex + 1;
      testHighIndex = testHighIndex + intervalSize;
      
      
      
  end
  
  errorRates = 100*(errorRates/fold);
  

  
end


%%% Plot example

% line plot example random data
%plot(10*rand(10,1))
%ylabel('accuracy')

% image plot
%imshow(mnist.train_images(:,:,3)) % plot the third image
