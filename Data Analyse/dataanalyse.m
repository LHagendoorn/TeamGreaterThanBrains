data = read_mixed_csv('ML_IN_PRACTICE/YELP/train.csv',',')

data = data(2:end,:)

counts = zeros(9,9);
countnr = zeros(9,1);
for i = data'
    numbers = regexprep(i(2,:),'[^\w'']','');
    numbers = numbers{:};
    for k = 1:length(numbers)
        nr = str2double(numbers(k));
        countnr(nr+1,1) = countnr(nr+1,1)+1;
        for l = k:length(numbers)
            nr2 = str2double(numbers(l));
            counts(nr+1,nr2+1) = counts(nr+1,nr2+1)+1;
        end
    end
end

for i = 1:9
    counts(i,:) = counts(i,:)./countnr(i,1);
end
counts

%count occurances sequences
dat = data(:,2);
a=unique(dat,'stable');
b=cellfun(@(x) sum(ismember(dat,x)),a,'un',0);
[a,b];



