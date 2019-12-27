import pulp
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy import distance as geo_dist

class subproblem(object):
    '''
    This class uses the PuLP package and COIN_CMD solver to solve an integer program (IP). From a pool
    of possible villages, the algorithm is in tandem trying to:
    a) Select the optimal N villages to expand the EG program into
    b) Cluster these N villages into K optimal clusters

    '''
    def __init__(self, centroids, data, OOSC, distance_cap, total_villages, village_cap, village_min, small_cluster_OOSC_min):

        self.centroids = centroids
        self.data = data
        self.n = len(data)
        self.k = len(centroids)
        self.OOSC = OOSC
        self.distance_cap = distance_cap
        self.total_villages = total_villages
        self.village_cap = village_cap
        self.village_min = village_min
        self.small_cluster_OOSC_min = small_cluster_OOSC_min

        self.create_model()

    def distances(self, assignment):
            '''
            Returns the distance in km between a village and a cluster center

            The index of the village is stored in assignment[0] and the index of the
            center is stored in assignemnt[1]
            '''
            return geo_dist.vincenty(self.data[assignment[0]], self.centroids[assignment[1]]).km

    def create_model(self):
        '''
        This method formulates the IP which is solved by the .solve method
        '''

        '''
        Lists to subscript each decision variable (i,j)
        Each tuple in list efers to the ith data point and j_th cluster

        Create sublists which reference real clusters and the fake cluster
        The fake cluster refers to cluster 0. Villages assigned to this cluster by the IP
        are not chosen for the EG program and don't count towards the objective function and constraints
        '''
        self.assignments = [(i, j) for i in range(self.n) for j in range(self.k)]
        self.assignments_real = [(i,j) for i in range(0, self.n) for j in range(1, self.k)]
        self.assignments_fake = [(i,0) for i in range(0, self.n)]

        '''Creates a dictionary of binary decision variables with assignments list acting as the keys
        These are the decision variables the IP is optimizing subject to an objective and constraints.
        The (i,j) decision variable is 1 if the ith village is assigned to the jth cluster and 0 otherwise
        '''
        self.y = pulp.LpVariable.dicts('data-to-cluster assignments',
                                  self.assignments,
                                  lowBound=0,
                                  upBound=1,
                                  cat=pulp.LpBinary)


        #Dictionary of additional decision variables involved in the formulation of the IF/THEN constraint
        self.z = pulp.LpVariable.dicts('IFTHEN', range(self.k), cat = pulp.LpBinary)

        #Initialize the IP model for the PuLP package
        self.model = pulp.LpProblem("Model for assignment subproblem", pulp.LpMaximize)

        '''
        Objective function: 
        Part A) Maximize the number of OOSC in the real clusters
        Part B) Minimize the village-centroid distances of real clusters
        Part C) Minimize the additional z decision variables to make the IF/THEN constraint work
        '''
        self.model += pulp.lpSum([(self.y[assignment]*(self.OOSC[assignment[0]] - .4*self.distances(assignment))) for assignment in self.assignments_real]) - .0001*pulp.lpSum([self.z[k] for k in range(1,self.k)]), 'Objective Function - Maximize number of OOSC assigned to real clusters'
        

        #--------Constraints---------#

        ''' Constraint 1: 
        Each village must be assigned to one cluster
        The algorithm chooses the villages which shouldn't be clustered by
        assigning them to cluster 0 (fake cluster)
        '''
        for i in range(self.n):
            self.model += pulp.lpSum([self.y[(i, j)] for j in range(self.k)]) == 1, "must assign point {}".format(i)


        '''Constraint 2: 
        Each village-centroid distance must be under distance_cap
        '''
        for assignment in self.assignments_real:
            self.model += self.y[assignment]*self.distances(assignment) <= self.distance_cap

        '''Constraint 3:
        Number of total villages assigned to real clusters is less than or equal to 
        the number EG wants to expand into (total_villages)
        
        Made this less than or equals to instead of equal to, to make sure algorithm isn't forced to make 
        bad clusters that hardly increase the number of OOSC reached while greatly increasing cluster 
        size/distances. Theoretically, if a good solution exists with the number of villages clustered
        equal to total villages, the algorithm should fufill this as an equality constraint since it is
        mainly trying to optimize the number of OOSC.

        Added print statement that checks whether solution clusters fewer than total_villages'''

        self.model += pulp.lpSum([self.y[assignment] for assignment in self.assignments_real]) <= self.total_villages

        '''
        Constraint 4:
        Number of villages in a cluster must not exceed village_cap 
        (Excluding the villages in the fake cluster)
        '''
        for j in range(1,self.k):
            self.model += pulp.lpSum([self.y[i,j] for i in range(self.n)]) <= self.village_cap
    

        '''
        Constraint 5:
        The following two constraints together model the IF/THEN logical statement: If the number of 
        villages in a cluster is less than or equal to village_min then the number of OOSC children in
        the cluster must be larger than small_cluster_OOSC_min

        If cluster k has less than village_min villages then the first constraint makes the z_k decision 
        variable equal to 1 and otherwise the z_k variable will be 0. When z_k is 1 the second constraint
        forces the umber of OOSC in the cluster to be greater than small_cluster_OOSC_min
        '''
        for j in range(1, self.k):
            self.model += ((self.village_min + 1) - pulp.lpSum([self.y[i,j] for i in range(self.n)]))/(self.village_min + 1) <= self.z[j]
            self.model += pulp.lpSum([self.y[i,j]*self.OOSC[i] for i in range(self.n)]) >= self.small_cluster_OOSC_min * self.z[j]


    def solve(self):
        '''Solve the IP which was formulated in the .create_model method
        Returns the optimal cluster assignment for each village
        '''
        #Solve IP and return the status of the solver
        self.status = self.model.solve(pulp.COIN_CMD())
        print("IP status", self.status)

        #Check how many clusters have 4 or fewer villages to see how IF/THEN constraint is changing clusters
        for k in range(1,self.k):
            print("1 if cluster {} has {} or fewer villages".format(k, self.village_min), self.z[k].value())

        #Turn binary decision variables into cluster IDs
        clusters = None
        if self.status == 1:
            print("Objective function value:", self.model.objective.value())
            clusters= [-1 for i in range(self.n)]
            for i in range(self.n):
                for j in range(self.k):
                    if self.y[(i, j)].value() > 0:
                        #Gives each data point a cluster ID according to the binary variables y_ij 
                        clusters[i] = j

        return clusters, self.status


'''
This is the end of the class subproblem which is constructing and solving the IP

Everything after this is its own function of which minsize_kmeans is the main function that
uses the IP as the cluster assignment step of a k-means clustering algorithm
'''

def initialize_centers(dataset, k):

    """
    sample k random datapoints as starting centers of clusters
    """
    ids = list(range(len(dataset)))
    random.shuffle(ids)
    return [dataset[id] for id in ids[:k]]


def compute_centers(clusters, dataset):
    '''
    1. Recalculates cluster_ids, shifting cluster_ids to get rid of clusters that had 
    no villages assigned to them

    2. Calculates centers of these clusters
    '''

    #List of unique current cluster IDs
    ids = list(set(clusters))
    c_to_id = dict()
    for counter, id_val in enumerate(ids):
        #Dictionary with keys = current_cluster_id and value = new_cluster_id
        c_to_id[id_val] = counter
    for counter, cluster_id in enumerate(clusters):
        #Reassigns old cluster_ids to correct new_cluster IDs using c_to_id dict
        clusters[counter] = c_to_id[cluster_id]

    #Calculate new cluster centers
    num_clust = len(ids)
    mean_lats = []
    mean_longs = []
    for k in range(num_clust):
        indices = np.where(np.asarray(clusters) == k)[0]
        subset = dataset[indices]
        mean_lats.append(np.mean(subset[:,0]))
        mean_longs.append(np.mean(subset[:,1]))

    cluster_centers = np.column_stack([mean_lats, mean_longs])
    
    return clusters, cluster_centers

def distance_calc(datapoint, center):
    #Returns the distance in km between a village and a cluster center
    return geo_dist.vincenty(datapoint, center).km

def kmeans_IP_run(dataset, merged_dataset, k, distance_cap, total_villages, village_cap, village_min, small_cluster_OOSC_min, column_name = "OOSCpredicted", max_iters=999):
    """
    Dataset - numpy matrix (or list of lists) of village coordinates
    Merged_dataset - Pandas dataframe of villages lat/long coordinates and predicted OOSC
    k: Number of clusters to make
    distance_cap: Max village-center distance allowed in cluster
    total_villages: Number of villages that should be assigned to real clusters for program expansion
    village_cap: Max number of villages allowed in clsuter
    village_min: If number of villages under this in cluster then put min on OOSC
    small_cluster_OOSC_min: If number of villages in cluster at or under village_min then enforce this min on OOSC
    column_name: Name of column corresponding to OOSC
    max_iters: If no convergence after this number of iterations, stop anyway

    Runs the a k-means algorithm with the typical cluster assignment step replaced by the IP
    """
    n = len(dataset)

    #Returns a list of k coordinates from the original dataset as the random initial centroids
    centers = initialize_centers(dataset = dataset, k = k)
    print("Centers", centers)
    #Creates a list of -1s of length n as the initial cluster assignments
    clusters = [-1] * n

    #Putting a limit on the number of iterations in the k-means algorithm
    cluster_all_dists = {}
    iteration_dist_avg = []
    iteration_OOSC_avg = []
    iteration_OOSC_total = []
    for ind in range(max_iters):
        #Creates new object m of type subproblem as defined by the subproblem class at start of this file
        #m object inherits all the methods defined in the subproblem class
        m = subproblem(centroids = centers, data = dataset, OOSC = np.array(merged_dataset[[column_name]].values), distance_cap = distance_cap, total_villages = total_villages, village_cap = village_cap, village_min = village_min, small_cluster_OOSC_min = small_cluster_OOSC_min)

        '''m.solve calls the pulp solve command and returns a list of the length of the data with a cluster 
        id for each datapoint if the pulp algorithm has found an optimal solution

        Where is m.create_model() called to actually set up the IP before solve is called?
        I think that is why it is called in self.init; it automatically formulates problem when the class is created
        '''
        clusters_, status_ = m.solve()
        while status_ == -3:
            print("-3 ERROR: REINITIALIZING CENTERS AND TRYING AGAIN")
            centers = initialize_centers(dataset = dataset, k = k)
            clusters = [-1] * n
            m = subproblem(centroids = centers, data = dataset, OOSC = OOSC, distance_cap = distance_cap, total_villages = total_villages, village_cap = village_cap, village_min = village_min, small_cluster_OOSC_min = small_cluster_OOSC_min)
            clusters_, status_ = m.solve()

        print("Iteration {}".format(ind))
        print("Clusters {}".format(clusters_))

        if not clusters_:
            return None, None

        '''Compute cluster allocations dropping clusters with no villages 
        and compute new cluster centers
        '''
        clusters_, centers = compute_centers(clusters_, dataset)

        '''Compute statistics at each iteration to see if algorithm is working
        Computed: 
        1. Avg number of OOSC across clusters to see if it is increasing
        2. Total number of OOSC across clusters to see if it is increasing
        3. Avg village-centroid distance across clusters to see if it is decreasing
        '''
        #Recalculate clusters since it might have changed after compute_centers
        if status_ != -3:
            k = len(centers)

        #Exclude the fake cluster for these stats
        if ind > 0:
            for j in range(1,k):
                #Extract villages in cluster k
                clusters_arr = np.asarray(clusters)
                village_indices_in_k = np.where(clusters_arr == j)[0]
                villages_in_k = merged_dataset.iloc[village_indices_in_k]

                #Calculate all distances between villages and cluster centroids
                villages_k_GPS = villages_in_k[["GPS_lat", "GPS_long"]]

                distances_villages = []
                for village in villages_k_GPS.values:
                    dist_center = distance_calc(datapoint = village, center = centers[j])
                    distances_villages.append(dist_center)

                cluster_all_dists["Cluster_{}".format(j)] = distances_villages

            #Calculate avg village-centroid distance across clusters    
            dist_list = list(cluster_all_dists.values())
            flat_dist_list = np.asarray([item for sublist in dist_list for item in sublist])
            # print(np.mean(flat_dist_list))
            iteration_dist_avg.append(np.mean(flat_dist_list))

            #Select villages not in fake cluster
            #Calculate avg OOSC in a cluster across clusters
            clusters_arr = np.asarray(clusters)
            village_indices_clustered = np.where(clusters_arr != 0)[0]
            villages_clustered = merged_dataset.iloc[village_indices_in_k]

            villages_OOSC = villages_clustered[["OOSCpredicted"]]
            iteration_OOSC_avg.append(np.mean(villages_OOSC.values))
            iteration_OOSC_total.append(np.sum(villages_OOSC.values))


        '''Stopping condition:
        K_means converges when no villages switch clusters

        In the first iteration clusters_ is just a list of -1's and after this is contains
        the previous iterations clustering allocations.'''

        converged = all([clusters[i]==clusters_[i] for i in range(n)])

        clusters = clusters_

        '''
        Once converged or max iterations has been reached, generate figures to check whether algorithm
        is converging well
        '''
        if converged:
            print("K-means converged")

            #Plot figures of statistics over iterations
            plt.close('all')
            fig, (ax1,ax2,ax3) = plt.subplots(3, 1, sharex = True)

            # Plot each graph, and manually set the y tick values
            ax1.plot(iteration_dist_avg)
            ax1.set_title("Village-centroid dist avg")

            ax2.plot(iteration_OOSC_avg)
            ax2.set_title("OOSC village avg")

            ax3.plot(iteration_OOSC_total)
            ax3.set_title("OOSC total")
            ax3.set_xlabel('Iteration')

            plt.subplots_adjust(hspace = .3)
            plt.savefig("Kmeans_functionality_figures/k{}_distcap{}_totvill{}_villcap{}".format(k, distance_cap, total_villages, village_cap))
            break
        elif ind == (max_iters - 1):
            #Plot figures of statistics over iterations
            plt.close('all')
            fig, (ax1,ax2,ax3) = plt.subplots(3, 1, sharex = True)

            # Plot each graph, and manually set the y tick values
            ax1.plot(iteration_dist_avg)
            ax1.set_title("Village-centroid dist avg")

            ax2.plot(iteration_OOSC_avg)
            ax2.set_title("OOSC village avg")

            ax3.plot(iteration_OOSC_total)
            ax3.set_title("OOSC total")
            ax3.set_xlabel('Iteration')

            plt.subplots_adjust(hspace = .3)
            plt.savefig("Kmeans_functionality_figures/k{}_distcap{}_totvill{}_villcap{}".format(k, distance_cap, total_villages, village_cap))
            break

    print("Final clusters:", clusters)

    return clusters, centers


def final_stats(clusters, centers, merged_dataset, expansion_goal):
    '''
    Takes in the final clusters and centers and returns 3 pandas dataframes of summary statistics

    Dataframe 1 - Cluster level:

    clust_num: Cluster ID
    max_dist: Max village-centroid distance in cluster
    mean_dist: Average village-centroid distance in clusters
    var_dist: Variance of village-centroid distances in clusters
    Avg_OOSC: Avg OOSC in villages in clsuters 
    Total_OOSC: Total OOSC in cluster
    Num_villages: Number of villages in cluster

    Dataframe 2 - Cluster level: All village-centroid distances in cluster

    Dataframe 3 - Cluster level: Number of OOSC in each village in cluster
    '''
    k = len(centers)

    cluster_all_dists = {}
    cluster_all_OOSC = {}
    max_dist_vals = np.zeros(k)
    avg_dist_vals = np.zeros(k)
    var_dist_vals = np.zeros(k)
    OOSC_means = np.zeros(k)
    OOSC_totals = np.zeros(k)
    num_villages_cluster = np.zeros(k)

    for j in range(k):
        #Extract villages in cluster k
        clusters_arr = np.asarray(clusters)
        village_indices_in_k = np.where(clusters_arr == j)[0]
        villages_in_k = merged_dataset.iloc[village_indices_in_k]
        num_villages_in_k = villages_in_k.shape[0]
        num_villages_cluster[j] = num_villages_in_k

        #Calculate all distances between villages and cluster centroids
        villages_k_GPS = villages_in_k[["GPS_lat", "GPS_long"]]

        distances_villages = []
        for village in villages_k_GPS.values:
            dist_center = distance_calc(datapoint = village, center = centers[j])
            distances_villages.append(dist_center)

        #Create Dataframe 2
        cluster_all_dists["Cluster_{}".format(j)] = distances_villages

        #Calculate some stats about village-centroid distances within clusters
        distances_villages = np.asarray(distances_villages)
        max_dist_vals[j] = np.max(distances_villages)
        avg_dist_vals[j] = np.mean(distances_villages)
        var_dist_vals[j] = np.var(distances_villages)

        #Calculate OOSC stats within clusters (Dataframe 3)
        villages_k_OOSC = villages_in_k[["OOSCpredicted"]]
        OOSC_means[j] = np.mean(villages_k_OOSC)
        OOSC_totals[j] = np.sum(villages_k_OOSC)
        cluster_all_OOSC["Cluster_{}".format(j)] = [item for sublist in villages_k_OOSC.values for item in sublist] 
    
    #Combine all the stats into a pandas dataframe to return Dataframe 1
    cluster_num = np.arange(k)
    stats_df = pd.DataFrame({"clust_num":cluster_num,"max_dist": max_dist_vals, "mean_dist":avg_dist_vals, "var_dist":var_dist_vals, "Avg_OOSC": OOSC_means, "Total_OOSC": OOSC_totals, "Num_villages": num_villages_cluster})

    #Check whether the number of villages clustered is less than the expansion goal
    total_num_villages = np.sum(stats_df["Num_villages"][1:])
    if total_num_villages < expansion_goal:
        print("Expansion goal not met. Expansion goal was {} and only {} villages were clustered".format(expansion_goal, total_num_villages))

    #Calculate total number of OOSC reached
    OOSC_overall_total = np.sum(stats_df["Total_OOSC"][1:])
    print("Number of OOSC children reached:", OOSC_overall_total)

    return stats_df, cluster_all_dists, cluster_all_OOSC