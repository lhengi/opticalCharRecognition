%pkg load statistics

% random seed
rng(0);

% setup data
D = csvread('iris.csv');
X_train = D(:, 1:2);
y_train = D(:, end); 

%set up set of m
m = [10 20 30 50];
errorRate = zeros(1,length(m));
for i = 1:length(m)
    errorRate(i) = myFunction(X_train,y_train,m(i));
    fprintf("Noise level: %i  Error Rate: %.2f\n",m(i),errorRate(i));
end

%modfiy original code and put in a function
function errorRate = myFunction(X_train, y_train,m)
    
    %randomly select which data point will be changed to noise
    seletedIndices = randperm(150,m);
    
    % change selected data points to noise
    for i = 1:m
       newClass =  randperm(3, 1);
       
       %make sure it's randomized
       while newClass == y_train(seletedIndices(i))
           newClass =  randperm(3, 1);
       end
       y_train(seletedIndices(i)) = newClass;
    end
    

    % setup meshgrid
    [x1, x2] = meshgrid(2:0.01:5, 0:0.01:3);
    grid_size = size(x1);
    X12 = [x1(:) x2(:)];

    % compute 1NN decision 
    n_X12 = size(X12, 1);
    decision = zeros(n_X12, 1);
    for i=1:n_X12    
        point = X12(i, :);

        % compute euclidan distance from the point to all training data
        dist = pdist2(X_train, point);

        % sort the distance, get the index
        [~, idx_sorted] = sort(dist);        
        % do 3NN on the grid
        class1 = 0;
        class2 = 0;
        class3 = 0;
        
        for k = 1:3
            if (y_train(idx_sorted(k)) == 1)
                class1 = class1 + 1;
            elseif (y_train(idx_sorted(k)) == 2)
                class2 = class2 + 1;
            else
                class3 = class3 + 1;
            end            
        end
        
        if (class1 == class2 && class2 == class3)
            pred = randperm(3, 1);
        elseif (class1 > class2 && class1 > class3)
            pred = 1;
        elseif (class2 > class1 && class2 > class3)
            pred = 2;
        else
            pred = 3;
        end
        decision(i) = pred;
    end
    
    %display(decision);
    
    

    % plot decisions in the grid
    figure(m);
    decisionmap = reshape(decision, grid_size);
    imagesc(2:0.01:5, 0:0.01:3, decisionmap);
    set(gca,'ydir','normal');

    % colormap for the classes
    % class 1 = light red, 2 = light green, 3 = light blue
    cmap = [1 0.8 0.8; 0.8 1 0.8; 0.8 0.8 1];
    colormap(cmap);

    % satter plot data
    hold on;
    scatter(X_train(y_train == 1, 1), X_train(y_train == 1, 2), 10, 'r');
    scatter(X_train(y_train == 2, 1), X_train(y_train == 2, 2), 10, 'g');
    scatter(X_train(y_train == 3, 1), X_train(y_train == 3, 2), 10, 'b');
    hold off;
    

    dataSize = length(X_train);
    dataDecision = zeros(dataSize,1);
    
    % compute the error rate for data
    for i = 1:dataSize
        cord = X_train(i,:);
        
        hideCordX = X_train;
        hideCordY = y_train;
        
        hideCordX(i,:) = [];
        hideCordY(i) = [];
        
        dist = pdist2(hideCordX,cord);
        [~, idx_sorted] = sort(dist);
        
        
        % do 3NN on data points
        class1 = 0;
        class2 = 0;
        class3 = 0;
        
        for k = 1:3
            if (y_train(idx_sorted(k)) == 1)
                class1 = class1 + 1;
            elseif (y_train(idx_sorted(k)) == 2)
                class2 = class2 + 1;
            else
                class3 = class3 + 1;
            end            
        end
        
        if (class1 == class2 && class2 == class3)
            pred = randperm(3, 1);
        elseif (class1 > class2 && class1 > class3)
            pred = 1;
        elseif (class2 > class1 && class2 > class3)
            pred = 2;
        else
            pred = 3;
        end
        dataDecision(i) = pred;
    end
    
    %display(dataDecision);
    %display(y_train);
    
    % compare with original y_train and the difference between 3NN and
    % original data set
    errorRate = mean(dataDecision ~=  y_train);
    
    
    %display(m)




end

