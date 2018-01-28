import csv
import numpy as np

class Clusters:
    '''
    Class that stores the properties of a cluster such as
    cluster_id, the members (guest_id) in that cluster and center of mass
    or the average of each column elements in that cluster
    '''
    def __init__(self, cluster_id, members, center_of_mass):
        self.cluster_id = cluster_id
        self.members = members
        self.center_of_mass = center_of_mass

class Agglomerative:
    '''
    Class that contains the entire process of agglomerative clustering
    Each task is carried out by an individual function
    '''
    def initialize_clusters(self, data):
        '''
        Function that forms initial clusters from the given data
        :param data: the actual dataset that is given as an input
        :return: a dictionary in which the keys are 'cluster_ids' and values are the objects of
        class 'Clusters'
        '''
        # initialize an empty dictionary
        clusters_dict = {}
        # add information to the Clusters object and add it to the dictionary
        for row in data:
            members = []
            # add members
            members.append(row[0])
            # add center of mass which is entire record except for guest_id
            center_of_mass = row[1:]
            # create a new object
            cl = Clusters(row[0], members, center_of_mass)
            # add the object to the dictionary
            clusters_dict[row[0]] = cl
        # return the dictionary
        return clusters_dict

    def calculate_minimum_distance(self, clusters_dict):
        '''
        Function that calculates the minimum distance between two clusters
        :param clusters_dict: the dictionary containing the objects of type 'Clusters'
        :return: the ids of 2 clusters between which the distance is minimum
        '''
        # initialize variables
        min1 = -1.0
        min2 = -1.0
        min_euclidean_distance = 9999999
        # iterate through each key in the dictionary
        for id1 in clusters_dict.keys():
            # get the center of mass of that cluster into a list to calculate the distance
            list1 = clusters_dict[id1].center_of_mass
            # convert the list to a numpy array
            list1 = np.array(list1)
            # iterate through each key in the dictionary
            for id2 in clusters_dict.keys():
                # this may cause wrong results
                # compute distance only if it has not been calculated previously
                if id1 < id2:
                    # get the center of mass of that cluster into a list to calculate the distance
                    list2 = clusters_dict[id2].center_of_mass
                    # convert the list to a numpy array
                    list2 = np.array(list2)
                    # calculate the euclidean distance using numpy function
                    euclidean_distance = np.linalg.norm(list1 - list2)
                    # keep track of the minimum distance and the cluster ids
                    if euclidean_distance < min_euclidean_distance:
                        min_euclidean_distance = euclidean_distance
                        min1 = id1
                        min2 = id2
                #print('Distance between ', data[id1][0], ' and ', data[id2][0], ' is ', euclidean_distance)
        #print('Min euc dist ', min_euclidean_distance, ' min1 ', min1, ' min2 ', min2)
        # return the ids
        return min1, min2

    def calculate_center_of_mass(self, members_count1, current_center_of_mass1, members_count2, current_center_of_mass2):
        '''
        Function that calculates the center of mass for a given set of records
        It uses the weighted average technique to calculate the new average or the
        new center of mass
        :param members_count1: used as a weight which will be multiplied to the center of mass of the first cluster
        :param current_center_of_mass1: center of mass of the first cluster
        :param members_count2: used as a weight which will be multiplied to the center of mass of the second cluster
        :param current_center_of_mass2: center of mass of the second cluster
        :return: return the centre of mass for a single cluster formed out of the two clusters
        provided as input
        '''
        #2D list that will be converted to 2D numpy array
        avg = []
        # add the center of mass of first cluster, once for each element in the cluster (weighted average)
        for num in range(0, members_count1):
            avg.append(current_center_of_mass1)
        # add the center of mass of second cluster, once for each element in the cluster (weighted average)
        for num in range(0, members_count2):
            avg.append(current_center_of_mass2)
        # convert 2D list to 2D numpy array to calculate the mean
        avg = np.array(avg)
        # return the center of mass for the new cluster
        return np.mean(avg, axis=0).tolist()

    def form_clusters(self, data):
        '''
        Function that forms clusters from individual elements
        :param data: the dataset that is provided as the input
        :return: None
        '''

        # initializing clusters
        stage = 1
        clusters_dict = self.initialize_clusters(data)
        # loop till we get only a single cluster
        while(len(clusters_dict)!=1):
            print('\nNumber of clusters: ', len(clusters_dict))
            # finding minimum distance between two clusters
            key1, key2 = self.calculate_minimum_distance(clusters_dict)

            # obtaining the values that are necessary to calculate the new center of mass
            members_count1 = len(clusters_dict[key1].members)
            members_count2 = len(clusters_dict[key2].members)
            current_center_of_mass1 = clusters_dict[key1].center_of_mass
            current_center_of_mass2 = clusters_dict[key2].center_of_mass

            # calculating the new center of mass for the merged clusters
            new_center_of_mass = self.calculate_center_of_mass(members_count1, current_center_of_mass1, members_count2, current_center_of_mass2)

            # store the center of mass on the element with minimum index
            clusters_dict[key1].center_of_mass = new_center_of_mass
            print(clusters_dict[key1].cluster_id, ' >>> ', clusters_dict[key1].members, ' merged with ')
            # change made here
            #clusters_dict[key1].members.append(key2)

            print(clusters_dict[key2].cluster_id, ' >>> ', clusters_dict[key2].members)

            # print the smaller cluster size
            if len(clusters_dict[key1].members) < len(clusters_dict[key2].members):
                print('Smaller cluster is ', key1,' size', len(clusters_dict[key1].members))
                #print('Stage ', stage, ' size ', len(clusters_dict[key1].members))
            elif len(clusters_dict[key1].members) > len(clusters_dict[key2].members):
                print('Smaller cluster is ', key2,' size', len(clusters_dict[key2].members))
                #print('Stage ', stage, ' size ', len(clusters_dict[key2].members))
            else:
                print('Equal size clusters', len(clusters_dict[key2].members),' ', len(clusters_dict[key2].members))
                #print('Stage ', stage, ' size ', len(clusters_dict[key2].members))

            # merging the members of two clusters
            for num in range(0, len(clusters_dict[key2].members)):
                clusters_dict[key1].members.append(clusters_dict[key2].members[num])

            # pop the cluster with the higher cluster_id
            clusters_dict.pop(key2)
            stage += 1

        print('\nNumber of clusters: ', len(clusters_dict))
        size = 0
        # print the elements of last cluster
        for key in clusters_dict.keys():
            print(clusters_dict[key].cluster_id, ' >>> ', clusters_dict[key].members)
            size = len(clusters_dict[key].members)
        print('Size of cluster: ', size)


    def main(self):
        '''
        main function which gets the data from the csv file
        :return:
        '''
        # path of the csv file
        path = 'C:/abc.csv'
        # read the file into a list
        data = list(csv.reader(open(path)))

        # casting from string to float
        for row in range(1,len(data)):
            for col in range(0, len(data[1])):
                data[row][col] = float(data[row][col])

        # removing headers
        data.pop(0)

        # forming clusters
        self.form_clusters(data)

# creating an object of type 'Agglomerative'
agg = Agglomerative()
# call to the main function
agg.main()
