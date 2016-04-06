
import pandas as pd
from datetime import datetime


class Aggregator:

    def __init__(self, Xdata, features, photo_order, trainSet=True):
        self.Xdata = Xdata
        self.trainSet = trainSet #true if X data is trainset, if testset, then false
        self.features = features
        self.photo_order = photo_order

    '''
    Gives the aggregated features per business (the Median, Q1, Q3, Min and Max)

    Xdata, features, photo_order all are dataFrames.

    -Uses the photo_order data to know which feature matches which photoID,
    -Uses the Xtrain data to know which business has which photoIDs

    Returns: nrOfBusinesses x nrOfFeatures x 5 list! (not dataframe)
    '''
    def get_aggregated_data(self):

        #initialize
        counter = 0
        businessIndex = 0
        allBusinessIDs = self.Xdata.business_id.unique()
        nrOfBusinesses = len(allBusinessIDs)
        nrOfFeatures = self.features.shape[1]

        #pre-allocate space for aggregated data
        aggregatedData = [0] * nrOfBusinesses

        #for each business
        for businessID in allBusinessIDs:
            #extract photoIDs of this business
            ff = self.Xdata[(self.Xdata.business_id == businessID)]
            photosIDs = ff['photo_id']
            nrOfPhotos = len(photosIDs)

            #pre-allocate space for feature vectors of this business
            businessData = [0] * nrOfPhotos

            #count how many photosIDS did not have a feature vector
            nrOfMissingPhotos = 0
            #keep track of index to place the feature vector
            photoIndex = 0

            #walk through the photoIDs
            for photoID in photosIDs:

                #find index of this photo in the features
                photo_ref = str(photoID) + ''.join('m.jpg')
                indx = self.photo_order.loc[self.photo_order[0] == photo_ref]

                #check whether photo has a feature vector. If so:
                if not(indx.empty):
                    #save the feature vector of this photo
                    ind = indx.index[0]
                    feats = self.features.loc[ind,:]
                    ff = feats.values.tolist()
                    businessData[photoIndex] = ff
                    photoIndex += 1

                else: ##count the missing photo
                    nrOfMissingPhotos += 1

            #remove rows of missing photos
            businessData = businessData[:-nrOfMissingPhotos][:]

            #compute and save aggregated data
            aggregatedData[businessIndex] = self.aggregate(businessData)
            businessIndex += 1

            #print progress
            counter = counter + 1
            if counter%100 == 0:
                print('Progress aggregation. Business ' + str(counter))
                print datetime.now()

        #save aggregatedData to csv files
        self.save_aggregated_data(aggregatedData)

        return aggregatedData

    '''
    Gives the Median, Q1, Q3, Min and Max of the given featureVectors

    featureVectors = list of size nrOfPhotos x nrOfFeatures

    Returns: list of size nrOfFeatures x 4
    '''
    def aggregate(self, featureVectors):
        df = pd.DataFrame(featureVectors)

        q1 = df.quantile(0.25)
        q2 = df.quantile(0.5)
        q3 = df.quantile(0.75)
        hoog = df.max()

        dinges = [q1, q2, q3, hoog]
        result = pd.concat(dinges, axis=0)

        return result.values

    '''Saves data in csv file '''
    def save_aggregated_data(self, data):

        df = pd.DataFrame(data)

        if self.trainSet:
            filename = 'Train'
        else:
            filename = 'Test'

        #save in csv file
        df.to_csv('./input/aggData' + filename + '.csv', index=False, header=None)



