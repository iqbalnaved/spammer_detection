clear;clc;
addpath('C:\Users\Iqbal\Documents\cs573\project\DeepLearnToolbox-master\DBN');
addpath('C:\Users\Iqbal\Documents\cs573\project\DeepLearnToolbox-master\NN');
addpath('C:\Users\Iqbal\Documents\cs573\project\DeepLearnToolbox-master\util');

load rbm_train_features;

rbm_train_x = cell2mat(rbm_features(:,2)); 
rbm_train_y = cell2mat(rbm_features(:,3));

load dbn_train_features;

dbn_train_x  = cell2mat(dbn_features(:,2)); 
dbn_train_y  = cell2mat(dbn_features(:,3));

load dbn_test_features

dbn_test_x  = cell2mat(dbn_features(:,2)); 
dbn_test_y  = cell2mat(dbn_features(:,3));

%% Train the 100-100 hidden unit DBN and use its weights to initialize a NN

rand('state',0)
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   1;
opts.batchsize = min(factor(size(rbm_train_x,1)));
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, rbm_train_x, opts);
dbn = dbntrain(dbn, rbm_train_x, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%unfold dbn to nn
outputsize = 2;
nn = dbnunfoldtonn(dbn, outputsize);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = min(factor(size(dbn_train_x,1)));
nn = nntrain(nn, dbn_train_x, dbn_train_y, opts);
[er, bad] = nntest(nn, dbn_test_x, dbn_test_y);
display(sprintf('error = %.2f%%', er*100));
% assert(er < 0.10, 'Too big error');