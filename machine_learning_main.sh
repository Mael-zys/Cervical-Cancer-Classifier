#############################################################################################################
# We can use "Grid" or "Randomized" for cv_mode


# 1. use every pixels as features and use pca to reduce dimension
# using pca
# python -u machine_learning_main.py --binary False --mask_mode True --pca_mode "pca" --num_components 200 --cv_mode "Grid"

# using spca
# python -u machine_learning_main.py --binary True --mask_mode True --pca_mode "spca" --num_components 200 --cv_mode "Grid"


# 2. use martin feature mode
# python -u machine_learning_main.py --binary False --mask_mode True --feature_mode "martin" --cv_mode "Grid"


# 3. use marina feature mode
# python -u machine_learning_main.py --classifier "PSO_SVM" --binary True --mask_mode True --feature_mode "marina" --cv_mode "Grid"


# 4. use sift-kmeans feature mode
# using sift
# python -u machine_learning_main.py --binary True --mask_mode False --feature_mode "sift_kmeans" --sift_orb "sift" --num_clusters 60 --cv_mode "Grid"

# using orb
# python -u machine_learning_main.py --binary True --mask_mode False --feature_mode "sift_kmeans" --sift_orb "orb" --num_clusters 60 --cv_mode "Grid"


# 5. use dong feature mode
# python -u machine_learning_main.py --classifier "SVM" --binary False --mask_mode True --feature_mode "dong" --cv_mode "Grid"


# 6. combine martin, marina and dong feature mode
python -u machine_learning_main.py --classifier "SVM" --binary True --mask_mode True --feature_mode "martin_marina_dong" --cv_mode "Grid"