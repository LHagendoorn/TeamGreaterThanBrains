

%inlezen business en photo ID   {photo id, businessid}
data = read_mixed_csv('YELP/train_photo_to_biz_ids.csv',',');
data = data(2:end,:);
photoid_biz = [data(:,1), data(:,2)];

%read images
%vals = cellarray : 1 row of images at 281 x 281, and one row above it with business
%ID's.
fnames = dir('YELP/test/*.jpg');
numfids = length(fnames);
vals = cell(1,numfids,2);
o = 1;
for K = 1:numfids
    a = imread(fnames(K).name);
    [x,y,z] = size(a);
    if x >= 281 && y >= 281
        %crop images to same size
        vals{1,o,1} = imcrop(a,[0,0,281,281]);
        x = strmatch(num2str(K),photoid_biz(:,1));
        biz = photoid_biz(x,2);
        vals{1,o,2} = biz;
        o = o+1;
    end
end
vals = vals(1,1:o-1,:); 




