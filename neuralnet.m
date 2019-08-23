clear
tic;
train_input = dlmread('dengue_features_train.csv',',',1,1);
train_label = dlmread('dengue_labels_train.csv',',',1,1);
n = 1000;
predict_labels(transpose(train_input(1:n,4:end)),transpose(train_label(1:n,3)), transpose(train_input(n+1:end,4:end)),transpose(train_label(n+1:end,3)),n);
%error = sum(abs((train_label(n+1:end,3) - pred_labels)))/(1456-n)
toc

function pred_labels = predict_labels(train_inputs,train_labels,test_input,test_label,n)
net = patternnet([200]);   % 3-layer neural network of 20,10,10 neurons in each layer
net.layers{1}.transferFcn = 'poslin';
%net.layers{2}.transferFcn = 'tansig';
%net.layers{3}.transferFcn = 'poslin';
net.performFcn = 'mae';
net.trainFcn = 'trainrp';
%net.performParam.regularization = 0.0001;
net = train(net,train_inputs,train_labels);
pred_labels = net(test_input);
error = sum(abs((test_label - pred_labels)))/(1456-n)
end