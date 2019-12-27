from Kmeans_IP_formulation import *
import numpy as np
import pandas as pd

#Load data
file = "../Data/MP_Final_Predictions_with_GPS_coordinates.xlsx"
df = pd.read_excel(file)

#Districts EG wants to expand into
districts = ["KHANDWA", "KHARGONE", "SHIVPURI"]

#The minimum number of OOSC the IP requires to be in small clusters for each district
OOSC_min_vals = [175, 230, 175]

'''
The below nested for loops run the Kmeans/IP clustering algorithm on each block in each of 
the districts in the districts array. For each block it outputs the results and summary
statistics into an excel file that is saved in the Excel_results/Kmeans_IP folder 
'''

for district_name, OOSC_min in zip(districts, OOSC_min_vals):
	print("DISTRICT:", district_name)
	#Extract data corresponding to district
	district = df[df.District == district_name]
	#Generate list of all unique block names in district
	blocks = district.Block.unique()
	#Extract blocks EG is not already working in
	blocks_already = ["KHALWA", "BHAGBANPURA", "NARVAR"]
	blocks_new = [block for block in blocks if block not in blocks_already]
	print("Blocks in {} EG isn't already working in:".format(district_name), blocks_new)

	for block_name in blocks_new:
		#Select the data for the block
		block = district[district.Block == block_name]
		num_villages_block = block.shape[0]
		print("Number of villages in {}:".format(block_name), num_villages_block)

		#Extracting and formatting data subset for the k-means/IP function
		data = np.asarray(block[["GPS_lat", "GPS_long"]].values)
		merged_data = block[["GPS_lat", "GPS_long", "OOSCpredicted"]]

		'''
		Calculate/set the parameters for the Kmeans/IP algorithm:
		Parameter 1 -  Number of villages to assign to a real cluster
		Parameter 2 - Number of real clusters to make
		Parameter 3 - Cap on village-centroid distance
		Parameter 4 - Cap on number of villages in cluster
		Parameter 5 - Minimum on number of villages in a cluster for small cluster IF/THEN constraint
		Parameter 6 - Minimum on number of OOSC in a cluster for small cluster IF/THEN constraint
		'''

		#-------- Calculating Parameter 1 ----------#
		block_total_OOSC = np.sum(block["OOSCpredicted"].values)
		print("Total number of OOSC in {}".format(block_name), block_total_OOSC)

		districts_select = ["KHANDWA", "KHARGONE", "SHIVPURI"]
		EG_districts = df.loc[(df["District"].isin(districts_select)) & (~df["Block"].isin(blocks_already))]
		total_OOSC_across_states = np.sum(EG_districts["OOSCpredicted"].values)
		print("Total number of OOSC across 3 EG states:", total_OOSC_across_states)

		proportion_OOSC = block_total_OOSC/total_OOSC_across_states
		print("Proportion of total OOSC that {} block makes up:".format(block_name), proportion_OOSC)

		expansion_goal = 1800
		block_expansion = min(num_villages_block,int(np.round(proportion_OOSC * expansion_goal)))
		print("{} block expansion goal:".format(block_name), block_expansion)

		#-------- Calculating Parameter 2 ----------#

		avg_OOSC_block = np.mean(block.OOSCpredicted)
		print("Average number of OOSC per village in {} block:".format(block_name), avg_OOSC_block)

		avg_OOSC_cluster = 150
		avg_villages_per_cluster_theory = avg_OOSC_cluster/avg_OOSC_block
		print("Average villages per cluster in {} block if each cluster has 150 OOSC on average".format(block_name), avg_villages_per_cluster_theory)

		if avg_villages_per_cluster_theory <= 4:
			print("WARNING: The algorithm might have trouble clustering in this block because the number of OOSC is large on average, so trying to expand into lots of villages")
		
		avg_villages_per_cluster = 6
		number_clusters = int(np.round((block_expansion/avg_villages_per_cluster) + 1))
		print("Number of clusters to make in block {}:".format(block_name), number_clusters)

		#-------- Setting Parameters 3-6 ----------#

		distance_cap = 20
		village_cap = 8
		village_min = 4
		small_cluster_OOSC_min = OOSC_min

		#Other algorithm specifications for printing file names
		block_name = block_name
		cluster_initialization = "Random"
		#Weight of objective function 
		dist_weight = .4

		#--------Running the K-means/IP algorithm on block ----------#
		final_clusters, final_cluster_centers = kmeans_IP_run(dataset = data, merged_dataset = merged_data, k = number_clusters, distance_cap = distance_cap, total_villages = block_expansion, village_cap = village_cap, village_min = village_min, small_cluster_OOSC_min = small_cluster_OOSC_min, column_name = "OOSCpredicted", max_iters=999)
		#Calculating stats about cluster output from algorithm
		stats, cluster_dists, cluster_OOSC_nums = final_stats(clusters = final_clusters, centers = final_cluster_centers, merged_dataset = merged_data, expansion_goal = block_expansion)


		'''Write block level cluster results/stats to an excel file
		Sheet 0: Parameters of K-means/IP algorithm that generated cluster results
		Sheet 1: Village level dataframe with cluster_ids
		Sheet 2: Cluster level dataframe with lat/long of cluster centers
		Sheet 3: Cluster level dataframe with stats about each cluster
		Sheet 4: Cluster level dataframe with a list of the village-centroid distances in each cluster
		Sheet 5: Cluster level dataframe with a list of the OOSC in each village
		''' 

		#--------Generating dataframe for Sheet 0 ----------#
		sheet0_df = pd.DataFrame({"total_OOSC_across_states":[total_OOSC_across_states],"block_total_OOSC": [block_total_OOSC], 
			"proportion_OOSC": [proportion_OOSC], "block_expansion": [block_expansion], "avg_villages_per_cluster": [avg_villages_per_cluster], 
			"number_clusters":[number_clusters], "distance_cap":[distance_cap],"village_cap":[village_cap], "village_min": [village_min],"cluster_initialization":[cluster_initialization]})
		
		#--------Generating dataframe for Sheet 1 ----------#
		sheet1_df = block[["District","Block", "Village", "VillageCode", "GPS_lat", "GPS_long", "OOSCpredicted"]]
		sheet1_df = sheet1_df.reset_index(drop = True)
		pandas_column = pd.DataFrame({'cluster_id': final_clusters})
		sheet1_df = sheet1_df.join(pandas_column)

		#--------Generating dataframe for Sheet 2 ----------#
		sheet2_df = pd.DataFrame(final_cluster_centers, columns = ["GPS_lat", "GPS_long"])


		#--------Generating dataframe for Sheet 4 ----------#
		values = list(cluster_dists.values())
		keys = list(cluster_dists.keys())
		sheet4_df= pd.DataFrame(data = values).T
		sheet4_df.columns = [keys]

		#--------Generating dataframe for Sheet 5 ----------#
		values = list(cluster_OOSC_nums.values())
		keys = list(cluster_OOSC_nums.keys())
		sheet5_df= pd.DataFrame(data = values).T
		sheet5_df.columns = [keys]

		#Writes each dataframe to a sheet of an excel file
		writer = pd.ExcelWriter("../Excel_results/Kmeans_IP/{}_{}_distweight{}_distcap{}_villcap{}_villagemin{}_small_cluster_OOSC_min{}.xlsx".format(district_name,block_name,dist_weight, distance_cap, village_cap, village_min, small_cluster_OOSC_min))
		sheet0_df.to_excel(excel_writer = writer, sheet_name = 'Algorithm_attributes')
		sheet1_df.to_excel(excel_writer = writer, sheet_name = 'Cluster_results')
		sheet2_df.to_excel(excel_writer = writer, sheet_name = "Cluster_centers")
		stats.to_excel(excel_writer = writer, sheet_name = "Cluster_stats")
		sheet4_df.to_excel(excel_writer = writer, sheet_name = "Cluster_dists")
		sheet5_df.to_excel(excel_writer = writer, sheet_name = "OOSC_numbers")
		writer.save()

