clear
tic;
train_input = dlmread('dengue_features_train.csv',',',1,1);
train_label = dlmread('dengue_labels_train.csv',',',1,1);
n = 1000;
num_pcs = 1;
pred_labels=predict_labels(train_input(1:n,:),train_label(1:n,:),train_input(n+1:end,:), num_pcs);
error = sum(abs((train_label(n+1:end,3) - pred_labels)))/(1456-n)
toc

function pred_labels = predict_labels(train_inputs,train_labels,test_inputs, num_pcs)
[coeff,score,latent,tsquared,explained,mu] = pca(train_inputs(:,4:end));
new_feat = score(:, 1:num_pcs);
test_inputs_mean_centered = test_inputs(:,4:end) - mu;
test_feat = test_inputs_mean_centered / coeff(:,1:num_pcs)';
mdl = fitrkernel(new_feat, train_labels(:,3));
pred_labels = predict(mdl, test_feat);
end