###Check whether cluster 0 should be dropped for any of these

import pandas as pd
from Systematic_postprocessing_formulation import *

#Set file to display all columns of pandas dataframe when print statement is called
pd.set_option('display.max_columns', None)

#-------------- Load and analyze output of K-means/IP algorithm ------------#

#Load original dataset to extract block names
file = "../Data/MP_Final_Predictions_with_GPS_coordinates.xlsx"
df = pd.read_excel(file)

#Compile algorithm output and stats from block level runs into one dataset
cluster_results_original, cluster_stats_original, dist_stats_original, OOSC_stats_original, cluster_centers_original = compile_original_results(original_df = df)

#Analyze results of algorithm
EG_districts = EG_subset(df)
analyze_output(EG_districts_ = EG_districts, cluster_stats_ = cluster_stats_original, dist_stats_ = dist_stats_original)
stats_figures(dist_stats_ = dist_stats_original, cluster_stats_ = cluster_stats_original, postprocessing_stage = 0)

#Generate map for results - Use jupyter notebook

#------------- Postprocessing Stage 1 --------------------#
'''
Dissolve clusters with 4 or less villages and assign these villages to closest cluster
'''
print("--------------- Posptrocessing Stage 1: Dissolving clusters with 4 or less villages and reassigning them to closest cluster ----------")
postprocessing_small(cluster_results_ = cluster_results_original, cluster_stats_ = cluster_stats_original, cluster_centers_ = cluster_centers_original)

#Generate stats for these postprocessed results
file = "Excel_results/Postprocessed1/Postprocessed1_dropsmall.xlsx"
cluster_results_1 = pd.read_excel(file, sheet_name = "Cluster_assignments")
clusters_dropped_df = pd.read_excel(file, sheet_name = "Clusters_dropped")
cluster_vals_dropped = list(clusters_dropped_df["Clusters_dropped"].values)
cluster_centers_1 = compute_centers(cluster_results_1)

cluster_stats_1, dist_stats_1, OOSC_stats_1 = recompute_final_stats(centers_ = cluster_centers_1, cluster_results_ = cluster_results_1, clusters_dropped = cluster_vals_dropped)
analyze_output(EG_districts_ = EG_districts, cluster_stats_ = cluster_stats_1, dist_stats_ = dist_stats_1)
stats_figures(dist_stats_ = dist_stats_1, cluster_stats_ = cluster_stats_1, postprocessing_stage = 1)

#Generate map for this iteration of postprocessing - Use jupyter notebook file

#------------- Postprocessing Stage 2 --------------------#
'''
Uncluster villages who are 5 or more km from their cluster centers and reassign them to closest cluster
Helps with overlapping villages
'''

print("--------------- Posptrocessing Stage 2: Unclustering villages who are 5 or more km from their cluster centers and reassigning them to closest cluster ----------")
postprocessing_far(cluster_results_ = cluster_results_1, centers_df = cluster_centers_1)

# #Generate stats for these postprocessed results
file = "Excel_results/Postprocessed2/Postprocessed2_far.xlsx"
cluster_results_2 = pd.read_excel(file, sheet_name = "Cluster_assignments")
cluster_centers_2 = compute_centers(cluster_results_2)

cluster_stats_2, dist_stats_2, OOSC_stats_2 = recompute_final_stats(centers_ = cluster_centers_2, cluster_results_ = cluster_results_2, clusters_dropped = cluster_vals_dropped)
analyze_output(EG_districts_ = EG_districts, cluster_stats_ = cluster_stats_2, dist_stats_ = dist_stats_2)
stats_figures(dist_stats_ = dist_stats_2, cluster_stats_ = cluster_stats_2, postprocessing_stage = 2)

# #Generate map for this iteration of postprocessing - Use jupyter notebook file

#------------- Postprocessing Stage 3 --------------------#
'''
Bring the number of clustered villages up to 1800 with two steps:
A. Add in unclustered villages with 15 or more OOSC (15 demarks the start of the right tail of the OOSC dist)
B. Add in remaining necessary villages by increasing distance to existing cluster centers'''

print("--------------- Posptrocessing Stage 3a: Adding in villages with 15 or more OOSC ----------")

postprocessing_addbyOOSC(cluster_results_ = cluster_results_2, centers_df_ = cluster_centers_2)
file = "Excel_results/Postprocessed3/Postprocessed3a_addOOSC.xlsx"
cluster_results_3a = pd.read_excel(file, sheet_name = "Cluster_assignments")
cluster_centers_3a = compute_centers(cluster_results_3a)

print("------ Posptrocessing Stage 3b: Adding in enough remaining villages to reach 1800 goal in order of distance to existing clusters----------")

postprocessing_addbydist(cluster_results_ = cluster_results_3a, centers_df_ = cluster_centers_3a)
file = "Excel_results/Postprocessed3/Postprocessed3b_addbydist.xlsx"
cluster_results_3b = pd.read_excel(file, sheet_name = "Cluster_assignments")
cluster_centers_3b = compute_centers(cluster_results_3b)

#Generate stats for these postprocessed results
cluster_stats_3, dist_stats_3, OOSC_stats_3 = recompute_final_stats(centers_ = cluster_centers_3b, cluster_results_ = cluster_results_3b, clusters_dropped = cluster_vals_dropped)
analyze_output(EG_districts_ = EG_districts, cluster_stats_ = cluster_stats_3, dist_stats_ = dist_stats_3)
stats_figures(dist_stats_ = dist_stats_3, cluster_stats_ = cluster_stats_3, postprocessing_stage = 3)

#Generate map for this iteration of postprocessing - Use jupyter notebook file

#------------- Postprocessing Stage 4 --------------------#
'''
Manually fix a few clusters'''

print("--------------- Posptrocessing Stage 4: Manually fixing a few clusters ----------")
cluster_results_new = cluster_results_3b.copy()

#Fix overlapping clusters
#Dissolve cluster 172 and reassign to either 211 or 212
cluster_results_new.loc[(cluster_results_new.Block == "BADARWAS") & (cluster_results_new.Village == "LAGDA"), "cluster_id"] = 211
cluster_results_new.loc[cluster_results_new.cluster_id == 172, "cluster_id"] = 212

# Move village in 234
cluster_results_new.loc[(cluster_results_new.Block == "PICHHORE") & (cluster_results_new.Village == "KAMALPUR"), "cluster_id"] = 195

# Move village in 277
cluster_results_new.loc[(cluster_results_new.Block == "SHIVPURI") & (cluster_results_new.Village == "SOOND"), "cluster_id"] = 241

#Appending cluster 172 to list of clusters that have been dropped in postprocessing
cluster_vals_dropped.append(172)
clusters_dropped_df = pd.DataFrame({"Clusters_dropped":cluster_vals_dropped})

writer = pd.ExcelWriter("Excel_results/Postprocessed4/Postprocessed_manual.xlsx")
cluster_results_new.to_excel(excel_writer = writer, sheet_name = 'Cluster_assignments')
clusters_dropped_df.to_excel(excel_writer = writer, sheet_name = 'Clusters_dropped')
writer.save()

#Generating statistics for this postprocessing stage
file = "Excel_results/Postprocessed4/Postprocessed_manual.xlsx"
cluster_results_4 = pd.read_excel(file, sheet_name = "Cluster_assignments")
clusters_dropped_df = pd.read_excel(file, sheet_name = 'Clusters_dropped')
cluster_vals_dropped = list(clusters_dropped_df["Clusters_dropped"].values)

cluster_centers_4 = compute_centers(cluster_results_4)

#Generate stats for these postprocessed results
cluster_stats_4, dist_stats_4, OOSC_stats_4 = recompute_final_stats(centers_ = cluster_centers_4, cluster_results_ = cluster_results_4, clusters_dropped = cluster_vals_dropped)
analyze_output(EG_districts_ = EG_districts, cluster_stats_ = cluster_stats_4, dist_stats_ = dist_stats_4)
stats_figures(dist_stats_ = dist_stats_4, cluster_stats_ = cluster_stats_4, postprocessing_stage = 4)

#------------- Postprocessing Stage 5 --------------------#
'''
Refix small clusters that were created during postprocessing
Dissolves clusters with 4 or less villages and less than 100 OOSC and reassigns these villages
to closest cluster
'''

print("--------- Posptrocessing Stage 5: Dissolving and reassigning clusters with 4 or fewer villages and less than 100 OOSC  ----------")
postprocessing_small_if_few(cluster_stats_ = cluster_stats_4, cluster_results_ = cluster_results_4, centers_df_ = cluster_centers_4, cluster_vals_dropped_ = cluster_vals_dropped)

#Generate stats for these postprocessed results
file = "Excel_results/Postprocessed5/Postprocessed5_smallfew.xlsx"
cluster_results_5 = pd.read_excel(file, sheet_name = "Cluster_assignments")
clusters_dropped_df = pd.read_excel(file, sheet_name = 'Clusters_dropped')
cluster_vals_dropped = list(clusters_dropped_df["Clusters_dropped"].values)
cluster_centers_5 = compute_centers(cluster_results_5)

cluster_stats_5, dist_stats_5, OOSC_stats_5 = recompute_final_stats(centers_ = cluster_centers_5, cluster_results_ = cluster_results_5, clusters_dropped = cluster_vals_dropped)
analyze_output(EG_districts_ = EG_districts, cluster_stats_ = cluster_stats_5, dist_stats_ = dist_stats_5)
stats_figures(dist_stats_ = dist_stats_5, cluster_stats_ = cluster_stats_5, postprocessing_stage = 5)

#------------- Renumber cluster IDs in final results --------------------#
'''
Renumber cluster IDs so that they are continuous accounting for dropped clusters
'''
print("--------- Posptrocessing done, Renumbering clusters so that cluster IDs are continuous ----------")

renumber_clusters(cluster_results_ = cluster_results_5)
#Generate stats for these postprocessed results
file = "Excel_results/Renumbered/Final_clusters.xlsx"
cluster_results_final = pd.read_excel(file, sheet_name = "Cluster_assignments")
cluster_centers_final = compute_centers(cluster_results_5)

cluster_stats_final, dist_stats_final, OOSC_stats_final = recompute_final_stats(centers_ = cluster_centers_final, cluster_results_ = cluster_results_final)
analyze_output(EG_districts_ = EG_districts, cluster_stats_ = cluster_stats_final, dist_stats_ = dist_stats_final)
stats_figures(dist_stats_ = dist_stats_final, cluster_stats_ = cluster_stats_final, postprocessing_stage = "final")

