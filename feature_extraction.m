clear;clc;
%%
train = 0; % 1=train,0=test

if train == 1
    spamfile = 'train_polluters_50k.txt';
    legitfile = 'train_legitimate_50k.txt';
else
    spamfile = 'test_polluters_50k.txt';
    legitfile = 'test_legitimate_50k.txt';    
end

fid = fopen(['C:\Users\Iqbal\Documents\cs573\project\data\' spamfile]);
data1 = textscan(fid, '%s\t%s\t%s\t%s', 'Delimiter','\t');
data1 = [data1{:,1} data1{:,3}]; % keep only userid and tweet

fid = fopen(['C:\Users\Iqbal\Documents\cs573\project\data\' legitfile]);
data2 = textscan(fid, '%s\t%s\t%s\t%s', 'Delimiter','\t');
data2 = [data2{:,1} data2{:,3}];

% user ids
spammer_list = unique(data1(:,1));
legit_userlist = unique(data2(:,1));

% combine 
data = [data1;data2];
clear data1 data2;
display('data loading finished..');

% Filter users with tweet freq = 1
[unique_data,~,ind] = unique(data(:,1)); 
freq_unique_data = histc(ind,1:numel(unique_data));
userid_list = unique_data(freq_unique_data==1); % list of users with 1 tweet
idx = find(ismember(data{1},userid_list));
for i = 1:2
    data{i}(idx)=[]; % delete rows 
end

stopwords = {'a';'able';'about';'across';'after';'all';'almost';'also';'am';'among';'an';'and';'any';'are';'as';'at';'be';'because';'been';'but';'by';'can';'cannot';'could';'dear';'did';'do';'does';'either';'else';'ever';'every';'for';'from';'get';'got';'had';'has';'have';'he';'her';'hers';'him';'his';'how';'however';'i';'if';'in';'into';'is';'it';'its';'just';'least';'let';'like';'likely';'may';'me';'might';'most';'must';'my';'neither';'no';'nor';'not';'of';'off';'often';'on';'only';'or';'other';'our';'own';'rather';'said';'say';'says';'she';'should';'since';'so';'some';'than';'that';'the';'their';'them';'then';'there';'these';'they';'this';'tis';'to';'too';'twas';'us';'wants';'was';'we';'were';'what';'when';'where';'which';'while';'who';'whom';'why';'will';'with';'would';'yet';'you';'your'};
% remove hashtag, @user, and URLs from all tweets, 
% remove stop words, perform stemming
for i = 1:length(data(:,1))
    data{i,2} = regexprep(data{i,2},'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)','');
    %convert tweet to wordlist
    wl = strsplit(data{i,2},' ');
    % convert to lowercase
    wl = cellfun(@lower, wl, 'UniformOutput', false);
    %remove empty strings
    wl(strcmp('',wl)) = [];
    %remove stop words
    idx = find(ismember(wl,stopwords));
    wl(idx) = [];
    % perform stemming on all words
    wl = cellfun(@porterStemmer, wl, 'UniformOutput', false);
    data{i,2} = wl;
end

display('data filtering complete...');

%% build vocabulary
if train == 1

    % collect all words from all tweets
    masterlist = [data{:,2}];

    % filter words with freq=1 from masterlist
    [unique_data,~,ind] = unique(masterlist); 
    freq_unique_data = histc(ind,1:numel(unique_data));
    word_list = unique_data(freq_unique_data==1); % list of users with 1 tweet
    idx = find(ismember(masterlist,word_list));
    masterlist(idx) = [];

    % Sort the words by frequency count.
    % retain the top k words with respect to information gain score. (k=1500 and k=1000 used)
    [unique_data,~,ind] = unique(masterlist); 
    freq_unique_data = histc(ind,1:numel(unique_data));
    [~,sortIdx]=sort(freq_unique_data,'descend');
    vocab = unique_data(sortIdx);

    m = round(mean(freq_unique_data)); 
    k = length(freq_unique_data(freq_unique_data>m)); 
    vocab = vocab(1:k);
    display(sprintf('feature vector size k=%d', k));
    save('vocab','vocab');
else
    load vocab;
end

display('creating features...');
% create binary features for RBM training
% checking all tweets of each user and putting 1 and 0 based on the presence and absense 
% of the word.
% create features for DBN
% checking all tweets of each user against vocabulary and put the frequency 
% of the occuracy of the word in that tweet.
userid_list = unique(data(:,1));
if train == 1
    rbm_features = cell(length(userid_list),3);
end
dbn_features = cell(length(userid_list),3);

for i = 1:length(userid_list)
    row_nums = find(ismember(data(:,1),userid_list(i)));
    userdata = data(row_nums,2);
    userdata = [userdata{:}];
    
    if ismember(userid_list{i}, spammer_list) == 1
        label = [1 0];
    else
        label = [0 1];
    end

    if train == 1
        binarylist = ismember(vocab, userdata);
        rbm_features{i,1} = userid_list{i};
        rbm_features{i,2} = double(binarylist);
        rbm_features{i,3} = label;
    end
    
    freqlist = zeros(1, length(vocab));
    vwords = vocab(ismember(vocab, userdata));
    for j = 1:length(vwords)
        list = strfind(vocab, vwords{j});
        indx = find(not(cellfun('isempty', list)));    
        freqlist(indx) = freqlist(indx) + 1;
    end
    dbn_features{i,1} = userid_list{i};
    dbn_features{i,2} = double(freqlist);   
    dbn_features{i,3} = label;
    
end

clear data;

display('saving features...');

if train == 1
    save ('rbm_train_features', 'rbm_features');
    save ('dbn_train_features', 'dbn_features');
else
    save ('dbn_test_features', 'dbn_features');
end
