

%voorbeeld - hier moet onze data
data = read_mixed_csv('YELP/train.csv',',');
data = data(2:end,:);


%input:
%Nx2 cellarray {businessname, labels}
%
%   businessname = 'sg00000'    [string]
%   labels = '0 1 2 3 4 5'      [string]

fid = fopen('output.csv','wt');
 if fid > 0
     fprintf(fid,'%s,%s\n','business_id','labels')
     for k = 1:size(data,1)
         fprintf(fid,'%s,%s\n',data{k,:});
     end
     fclose(fid);
 end
 outdata = read_mixed_csv('output.csv',',')
