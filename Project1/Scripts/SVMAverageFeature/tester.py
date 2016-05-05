

import pandas as pd

data = [[12, 345],[12,348],[14,500],[14,540]]

validPhotoIds = pd.DataFrame(data,columns=['business_id', 'photo_id'])

validPhotoIds.photo_id = validPhotoIds.photo_id.map(lambda x: str(x))

validPhotoIds.photo_id.astype(str)

ding = [['344', 0.0, 0.45], ['345', 0.2, 0.56], ['348', 0.34, 0.56], ['500', 3.4, 2.5], ['501',2.3, 4.5],['540',0.6,0.4]]

allData = pd.DataFrame(ding,columns=['photo_id', 'feature1', 'feature2'])

validationFeatures = pd.merge(validPhotoIds, allData, left_on='photo_id', right_on='photo_id')

print 'klaar'