import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

df = pd.read_csv('Pokemon.csv')

types = {} # Collect Type 1s
for val in df['Type 1']:
    if val in types: types[val] += 1
    else: types[val] = 1

df_stats = df.drop(columns= ['#', 'Type 1', 'Type 2', 'Total', 'Generation', 'Legendary'])
df = df_stats.drop(columns= ['Name'])

steps = [
    ('scale', MinMaxScaler()),
    ('cluster', KMeans())
]

pipe = Pipeline(steps= steps)

global_preds = {}

for t in types:
    cluster_max = min(types[t], 15) # Can't have more clusters then pokemon of type

    grid = {
        'cluster__n_init': ['auto'], # Stops warnings
    }

    best_score = 0
    num_clusters = 0
    best_pred = []

    print('\n' + str(t) + '\n-----------')

    for n_clusters in range(2, cluster_max):
        pipe.set_params(cluster__n_clusters = n_clusters) # Set n_clusters

        search = GridSearchCV(estimator= pipe, param_grid= grid, n_jobs= -1).fit(df)
        pred = search.predict(df)
        
        score = silhouette_score(df, pred)

        print(str(n_clusters) + ' clusters: ' + str(score))

        if score > best_score: # Track best amount of clusters
            best_score = score
            num_clusters = n_clusters
            best_pred = pred

    print('best number of clusters: ' + str(num_clusters))
    print('best score: ' + str(best_score))
    global_preds[t] = [num_clusters, best_pred]

df_pred = df_stats

for t in types:
    print('\n' + str(t) + '\n-----')

    df_pred['pred'] = global_preds[t][1]

    for i in range(0, global_preds[t][0]): # Each Cluster
        print('Cluster ' + str(i))

        df_final = df_pred[df_pred['pred'] == i]
        print(df_final.drop(columns= ['pred']))

        print('Mean HP: ' + str(df_final['HP'].mean()))
        print('Mean Attack: ' + str(df_final['Attack'].mean()))
        print('Mean Defense: ' + str(df_final['Defense'].mean()))
        print('Mean Sp. Atk: ' + str(df_final['Sp. Atk'].mean()))
        print('Mean Sp. Def: ' + str(df_final['Sp. Def'].mean()))
        print('Mean Speed: ' + str(df_final['Speed'].mean()) + '\n')
