import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as pplot
from mpl_toolkits.mplot3d import axes3d


def main():
    # path of the csv file
    path = 'C:\\abc.csv'
    # read the file into a list
    input_file = list(csv.reader(open(path)))
    # initialize an empty list
    data = []
    # casting from string to float
    for row in range(0, len(input_file)):
        for col in range(0, len(input_file[1])):
            input_file[row][col] = float(input_file[row][col])

    get_eps(input_file)
    dbscan(input_file)


def get_eps(data):
    '''
    This function plots the 'n' th neighbor of each point in the data where 0 < n < 11
    The eps value can be calculated from the graph
    :param data: input file
    :return: None
    '''
    # k value ranges from 1 to 10
    for k in range(1, 11):
        # list containing the distance of kth neighbor
        eps = []
        # for every row in data
        for row2 in data:
            # list containing the distances of all the points
            neighbors = []
            # converting to numpy array
            list2 = np.array(row2)
            # for every row in data
            for row1 in data:
                # converting to numpy array
                list1 = np.array(row1)
                # calculating the euclidean distance
                euclidean_distance = np.linalg.norm(list1 - list2)
                # adding the distance to the list
                neighbors.append(euclidean_distance)
            # sorting the distances
            neighbors.sort()
            # adding the distance of kth neighbor to the list
            eps.append(neighbors[k])
        # sorting the distances
        eps.sort()
        # plotting the distances
        pplot.plot(range(0, len(data)), eps)
    # labelling the graph
    pplot.xlabel('Data points')
    pplot.ylabel('Distance')
    pplot.title('Data points VS Distance of \'k\'th(1-10) neighbor')
    pplot.legend(range(1, 11))
    pplot.show()


def get_neighbors(row, data, eps):
    '''
    This function calculated the neighboring points i.e. the points within the eps distance of the point
    :param row: a row or a point from the data whose neighbors are to be found out
    :param data: the input file
    :param eps: the eps value calculated from the above function
    :return: a list of indices of the neighboring points
    '''
    # list of neighbors within the eps range
    neighbors = []
    # renaming for convenience
    list2 = row
    # convert the list to a numpy array
    list2 = np.array(list2)

    # for each index in data
    for num in range(0, len(data)):
        # renaming
        list1 = data[num]
        # convert the list to a numpy array
        list1 = np.array(list1)
        # calculate the euclidean distance using numpy function
        euclidean_distance = np.linalg.norm(list1 - list2)
        # if distance is less than eps value
        if euclidean_distance < eps:
            # add the index of that point to the neighbors list
            neighbors.append(num)
    # return the list
    return neighbors


def add_neighboring_points(num, neighbors, new_cluster_list, eps, min_pts, clusters, visited, data):
    '''

    :param num: the index of the data point
    :param neighbors: the neighbors of num
    :param new_cluster_list: new list to add elements to the new cluster
    :param eps: eps value calculated
    :param min_pts: minimum number of points that should exist in a cluster
    :param clusters: 2D list of clusters
    :param visited: list containing the indices of the visited data points
    :param data: input file
    :return: None
    '''
    # add the index to new cluster
    new_cluster_list.append(num)
    # for each neighboring point
    for num1 in neighbors:
        # if not present in visited list
        if num1 not in visited:
            # add it to the visited list
            visited.append(num1)
            # get its neighbors
            non = get_neighbors(data[num1], data, eps)
            # if length is greater than or equal to min_pts
            if len(non) >= min_pts:
                # for each element in neighbors
                for neb in non:
                    # if not in neighbors list
                    if neb not in neighbors:
                        # add it to the neighbors list
                        neighbors.append(neb)

        # initialize a flag
        flag = False
        # for each cluster in the clusters list
        for cl in clusters:
            # if the neighbor is in the cluster
            if num1 in cl:
                # mark the flag as True i.e. it is already a part of the cluster
                flag = True
                # so no need to add the point to any cluster
                break
        # if point not in any of the clusters
        if flag == False:
            # add that point to the cluster list
            new_cluster_list.append(num1)

    # append the new cluster list to the 2D clusters list
    clusters.append(new_cluster_list)


def calculate_center_of_mass(clusters, data):
    '''
    This function calculated the center of mass of each cluster
    :param clusters: 2D list of clusters
    :param data: input file
    :return: A list containing all the center of masses
    '''
    # a 2D list that will contain center of mass for each cluster
    center_of_mass_list = []
    # for each cluster in cluster list
    for cl in clusters:
        # create a new new list to store all the data points corresponding to the indices
        center_of_mass = []
        # for each element in the cluster
        for element in cl:
            # add all the elements in that cluster to the list
            center_of_mass.append(data[element])
        # converting to a numpy array
        com_array = np.array(center_of_mass)
        # calculating the mean
        com = np.mean(com_array, axis=0)
        # rounding off to 2 decimal places
        com = np.round(com, 2).tolist()
        # adding the center of mass to the center of mass list
        center_of_mass_list.append(com)
    # returning the list
    return center_of_mass_list


def plot_clusters(clusters, data):
    '''
    function that plots clusters in 3 dimensions
    :param clusters: generated clustes
    :return: None
    '''
    # colors used to represent different clusters
    colors = ['blue', 'red', 'green', 'yellow', 'magenta', 'black', 'grey', 'brown', 'orange', 'pink', 'cyan']
    # markers used to represent different clusters
    sym = ['^', 'o', '*', '+', 'x']
    fig = pplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    # cluster number
    num = 0
    for cl in clusters:
        # selecting the color
        c = colors[num % 10]
        # selecting the symbol
        s = sym[num % 4]
        for index in cl:
            row = data[index]
            # using a scatter plot
            ax.scatter(row[0], row[1], row[2], color=c, marker=s)
        num += 1
    pplot.show()


def dbscan(data):
    '''
    This function calculated the clusters based on their density
    :param data: input file
    :return: None
    '''

    # initializing the varibles
    # epsilon value
    eps = 0.61
    # minimum number of points that should exist in a cluster or else consider noise
    min_pts = 5
    # 2D list that will contain the list of points in each cluster
    clusters = []
    # list that will contain the indices of the visited data points
    visited = []
    # list that will contain the indices of the points that are considered as noise
    noise = []
    # for each index in the data
    for num in range(0, len(data)):
        # if the index is not in visited list
        if num not in visited:
            # add the index to the visited list
            visited.append(num)
            # get the neighbors of that point
            neighbors = get_neighbors(data[num], data, eps)
            # if the length of neighbors list is less than min_pts
            if len(neighbors) < min_pts:
                # add the index in the noise list
                noise.append(num)
            else:
                # create a new cluster list
                new_cluster_list = []
                # add the neighboring points of each cluster element
                add_neighboring_points(num, neighbors, new_cluster_list, eps, min_pts, clusters, visited, data)

    # calculate the center of mass of all the clusters
    center_of_mass_list = calculate_center_of_mass(clusters, data)
    # add columns cluster_no, number of elements and center of mass in a dataframe
    df = pd.DataFrame(columns=['cluster no', 'no of elements', 'center of mass'])
    # for all clusters
    for num in range(0, len(clusters)):
        # add the data to the dataframe
        df.loc[num] = [num, len(clusters[num]), center_of_mass_list[num]]

    # sort the dataframe by number of elements in ascending order
    df = df.sort_values(by='no of elements', ascending=True)
    # print the dataframe
    print(df)
    print('Noise points: ', len(noise))

    # 3D plot of the clusters
    plot_clusters(clusters, data)


main()
