%% Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance
%% This file is modified from pmtk3.googlecode.com

load('mnistData');

% set training & testing 

trainndx = 1:10000; 



trainingLength = [100 500 1000 2500 5000 7500 10000];

for i = 1:length(trainingLength)
  errs(i) = myFunction(trainingLength(i),mnist);
end

plot(trainingLength,errs);
xlabel('Train Length');
ylabel('Error rate');

% set up a function to train data in different length
function errorRate = myFunction(trainLen,mnist)
  % setup train data indices
  trainndx = 1:trainLen; 
  
  
  testndx =  1:10000; 
  ntrain = length(trainndx);
  ntest = length(testndx);
  Xtrain = double(reshape(mnist.train_images(:,:,trainndx),28*28,ntrain)');
  Xtest  = double(reshape(mnist.test_images(:,:,testndx),28*28,ntest)');

  ytrain = (mnist.train_labels(trainndx));
  ytest  = (mnist.test_labels(testndx));

  % Precompute sum of squares term for speed
  XtrainSOS = sum(Xtrain.^2,2);
  XtestSOS  = sum(Xtest.^2,2);

  % fully solution takes too much memory so we will classify in batches
  % nbatches must be an even divisor of ntest, increase if you run out of memory 
  if ntest > 1000
    nbatches = 50;
    else
    nbatches = 5;
  end
  batches = mat2cell(1:ntest,1,(ntest/nbatches)*ones(1,nbatches));
  ypred = zeros(ntest,1);

  % Classify
  for i=1:nbatches    
      dst = sqDistance(Xtest(batches{i},:),Xtrain,XtestSOS(batches{i},:),XtrainSOS);
      [junk,closest] = min(dst,[],2);
      ypred(batches{i}) = ytrain(closest);
  end
  % Report

  errorRate = mean(ypred ~= ytest);
  fprintf('Data Length: %i  Error Rate: %.2f%%\n',trainLen,100*errorRate);
  
end

%%% Plot example

% line plot example random data
%plot(10*rand(10,1))
%ylabel('accuracy')

% image plot
%imshow(mnist.train_images(:,:,3)) % plot the third image
