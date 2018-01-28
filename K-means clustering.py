import csv
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d


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

class K_means:
    '''
    Class that contains the entire process of kmeans clustering
    Each task is carried out by an individual function
    '''
    def initialize_clusters(self, data, k):
        '''
        Function that forms initial clusters from the given data
        :param data: the actual dataset that is given as an input
        :return: a dictionary in which the keys are 'cluster_ids' and values are the objects of
        class 'Clusters'
        '''
        # generates random k values from dataset
        random_points = random.sample(range(len(data)), k)
        data = [data[row] for row in random_points]

        # initialize an empty dictionary
        clusters_dict = {}
        cluster_id = 0
        # add information to the Clusters object and add it to the dictionary
        for row in data:
            #print(row)
            members = []
            # add members
            members.append(row)
            # add center of mass which is entire record except for guest_id
            center_of_mass = row
            # create a new object
            cl = Clusters(cluster_id, members, center_of_mass)
            # add the object to the dictionary
            clusters_dict[cluster_id] = cl
            # increment value of cluster_id
            cluster_id += 1
        # return the dictionary with k randomly chosen centroids
        return clusters_dict

    def adding_points_to_clusters(self, clusters_dict, data):
        '''
        Function that adds each data point to the clusters
        :param clusters_dict: the dictionary containing the objects of type 'Clusters'
        :return: the ids of 2 clusters between which the distance is minimum
        '''
        # for each row in the data
        for row in data:
            # initialize variables
            min_distance_cluster_id = -1
            min_euclidean_distance = 9999999
            # get the center of mass of that cluster into a list to calculate the distance
            list2 = row
            # convert the list to a numpy array
            list2 = np.array(list2)
            # iterate through each key in the dictionary
            for id in clusters_dict.keys():
                # get the center of mass of that cluster into a list to calculate the distance
                list1 = clusters_dict[id].center_of_mass
                # convert the list to a numpy array
                list1 = np.array(list1)
                # calculate the euclidean distance using numpy function
                euclidean_distance = np.linalg.norm(list1 - list2)
                # keep track of the minimum distance and the cluster ids
                if euclidean_distance < min_euclidean_distance:
                    min_euclidean_distance = euclidean_distance
                    min_distance_cluster_id = id

            # assigning the point to the nearest centroid or cluster
            clusters_dict[min_distance_cluster_id].members.append(row)
            # if len(clusters_dict)==6:
            #     for id in clusters_dict.keys():
            #         print('Cluster ', id, ': ', len(clusters_dict[id].members))
            #print('Row ', row, ' >>> ', min_distance_cluster_id)
        # return the dictionary
        return clusters_dict

    def calculate_center_of_mass(self, clusters_dict):
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

        # add the center of mass of first cluster, once for each element in the cluster (weighted average)
        for id in clusters_dict.keys():
            #2D list that will be converted to 2D numpy array
            avg = []
            # add each element in the cluster to a 2D list
            for member in clusters_dict[id].members:
                avg.append(member)

            # convert 2D list to 2D numpy array to calculate the mean
            avg = np.array(avg)

            # return the center of mass for the new cluster
            com = np.mean(avg, axis=0).tolist()
            com = np.round(com, 0)
            #print('COM for cluster ', id, ' is ', com)
            clusters_dict[id].center_of_mass = com

        return clusters_dict

    def change_in_centroid_position(self, cluster_dict, new_cluster_dict):
        '''
        this method checks for the change in old and new center of mass
        :param cluster_dict: input data
        :param new_cluster_dict: new calculated center of mass
        :return: whether both of the above are same
        '''
        if len(new_cluster_dict) == 0 or len(cluster_dict) == 0:
            return True

        # compare each element from both the dictionaries
        for id in new_cluster_dict.keys():
            for index in range(0, len(cluster_dict[id].center_of_mass)):
                if cluster_dict[id].center_of_mass[index] != new_cluster_dict[id].center_of_mass[index]:
                    return True
        return False

    def print_dict(self, clusters_dict):
        '''
        function to print the dictionary
        :param clusters_dict: dictionary containing clusters
        :return: None
        '''
        for id in clusters_dict.keys():
            print('>>>>>>>>>>>Id: ', id)
            print('------------Members------------ ')
            print(clusters_dict[id].members)
            print('Center of mass ', clusters_dict[id].center_of_mass)


    def form_clusters(self, data, k):
        '''
        Function that forms clusters from individual elements
        :param data: the dataset that is provided as the input
        :return: sse
        '''
        iteration = 1
        # initialize clusters
        clusters_dict = self.initialize_clusters(data, k)
        # dictionary to store the old version of center of mass
        old_clusters_dict = {}
        # dictionary that will store new calculated center of mass
        new_clusters_dict = copy.deepcopy(clusters_dict)
        # while old and new centroids are not same
        while(self.change_in_centroid_position(old_clusters_dict, new_clusters_dict)):
            old_clusters_dict = copy.deepcopy(new_clusters_dict)
            # add points to the clusters
            clusters_dict = self.adding_points_to_clusters(new_clusters_dict, data)
            # calculate center of mass
            new_clusters_dict = self.calculate_center_of_mass(clusters_dict)
            iteration += 1
        # calculate sum of squared errors
        sse = self.calculate_sse(new_clusters_dict)
        print('K = ', k, ' SSE = ',sse)
        #self.plot_clusters(new_clusters_dict)
        return sse


    def calculate_sse(self, cluster_dict):
        '''
        calculates the sse for given clusters
        :param cluster_dict: generated clusters
        :return:
        '''
        sse = -1
        for id in cluster_dict.keys():
            com = np.array(cluster_dict[id].center_of_mass)
            #print('Center of mass', com)
            for member in cluster_dict[id].members:
                temp = np.array(member)
                #print(temp)
                sse = sse + ((temp - com)**2).sum()
        return sse

    def plot_sse_vs_k(self, sse, k):
        '''
        function that plots the graph of sse vs K
        :param sse: sum of squared errors
        :param k: value of K
        :return: None
        '''
        plt.plot(k, sse)
        plt.show()

    def plot_clusters(self, cluster_dict):
        '''
        function that plots clusters in 3 dimensions
        :param cluster_dict: generated clustes
        :return: None
        '''
        # colors used to represent different clusters
        colors = ['blue', 'red', 'green', 'yellow', 'magenta', 'black', 'grey', 'brown', 'orange', 'pink', 'cyan']
        # markers used to represent different clusters
        sym = ['^', 'o', '*', '+', 'x']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for id in cluster_dict.keys():
            c = colors[id%10]
            s = sym[id%4]
            for row in cluster_dict[id].members:
                ax.scatter(row[0], row[1], row[2], color=c, marker=s)
        plt.show()

    def remove_noise(self, data):
        '''
        function that removes the noise if any
        :param data: input data
        :return: None
        '''
        for row in range(0,len(data)):
            count = 0
            x1 = data[row][0]
            y1 = data[row][1]
            z1 = data[row][2]
            while count<10:
                for inner_row in range(0,len(data)):
                    x2 = data[inner_row][0]
                    y2 = data[inner_row][1]
                    z2 = data[inner_row][2]
                    #print(x1, y1, z1, x2, y2, z2)
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
                    #print(dist)
                    if dist > 1:
                        count += 1
            if count < 9:
                #print('Row ', row, ' removed')
                data.remove(row)

    def main(self):
        '''
        main function which gets the data from the csv file
        :return:
        '''
        # path of the csv file
        path = 'C:/abc.csv'
        # read the file into a list
        input_file = list(csv.reader(open(path)))
        # initialize an empty list
        data = []
        negative = True
        # casting from string to float
        for row in range(1,len(input_file)):
            negative = False
            for col in range(0, len(input_file[1])):
                input_file[row][col] = float(input_file[row][col])
                if input_file[row][col] < 0:
                    negative = True
            # add the record to data if it does not have any negative values
            if not negative:
                data.append(input_file[row])


        # removing headers
        data.pop(0)
        # remove noise
        self.remove_noise(data)
        print(len(data))
        # forming clusters for multiple values of k
        #for iter in range(0, 6):
        sse = []
        for k in range(2, 11):
            sse.append(self.form_clusters(data, k))
        self.plot_sse_vs_k(sse, list(range(2, 11)))
        #plt.show()

# creating an object of type 'K_means'
kmeans = K_means()
# call to the main function
kmeans.main()
