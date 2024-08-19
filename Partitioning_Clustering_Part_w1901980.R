install.packages("readxl")
install.packages("NbClust")
install.packages("factoextra")
install.packages("cluster")

library(readxl)
library(NbClust)
library(factoextra)
library(cluster)

# SUBTASK 1

# Load the dataset
data <- read_excel("C:/Users/HP 15 - CS1032TX/Desktop/Whitewine_v6.xlsx")

# Separate features (attributes)
data <- data[, 1:11]

# Scale data
scaled_data <- scale(data)

# Visualize data before outlier removal
boxplot(scaled_data)

# Detect Outliers

# Calculate the quartiles for each column
q1 <- apply(scaled_data, 2, quantile, probs = c(0.25))
q3 <- apply(scaled_data, 2, quantile, probs = c(0.75))
iqr <- q3 - q1

# Calculate the upper and lower limits
upper <- q3 + 2.5 * iqr
lower <- q1 - 2.5 * iqr

# Identify the outliers
outliers <- apply(scaled_data, 2, function(x) x < lower | x > upper)

# Remove the outliers
data_no_outliers <- scaled_data[!apply(outliers, 1, any),]

# Visualize data after outlier removal
boxplot(data_no_outliers)

# Determine the Number of Cluster Centers

set.seed(42)

# NBclust
nb_clusters <- NbClust(data_no_outliers, distance = "euclidean", min.nc = 2, 
                       max.nc = 10, method = "kmeans", index = "all")

# Elbow method
fviz_nbclust(data_no_outliers, kmeans, method = "wss")

# Silhouette method
fviz_nbclust(data_no_outliers, kmeans, method = "silhouette")

# Gap static method
fviz_nbclust(data_no_outliers, kmeans, method = "gap_stat")

# K Means

# Clustering using kmeans algorithm
kmeans_model <- kmeans(data_no_outliers, centers = 2)

kmeans_model

fviz_cluster(kmeans_model, data = data_no_outliers, palette = c("#2E9FDF", "#DB4035", "#E7B800", "#b800e7"),
             geom = "point",
             ellipse.type = "convex",
             ggtheme = theme_bw())


wss <- kmeans_model$tot.withinss
bss <- kmeans_model$betweenss
tss <- wss + bss
bss_ratio <- bss / tss

wss
bss
tss
bss_ratio

# Silhouette
sil <- silhouette(kmeans_model$cluster, dist(data_no_outliers))
fviz_silhouette(sil)


# SUBTASK 2

# PCA Analysis
pca_result <- prcomp(data_no_outliers, center = TRUE, scale = TRUE)
summary(pca_result)

# The cumulative proportion exceeds 85% after PC7

# transformed dataset
data_transformed <- as.data.frame(-pca_result$x[,1:7])
head(data_transformed) 

View(data_transformed)

# NBclust
nb_clusters <- NbClust(data_transformed, distance = "euclidean", min.nc = 2, 
                       max.nc = 10, method = "kmeans", index = "all")

# Elbow method
fviz_nbclust(data_transformed, kmeans, method = "wss")

# Silhouette method
fviz_nbclust(data_transformed, kmeans, method = "silhouette")

# Gap static method
fviz_nbclust(data_transformed, kmeans, method = "gap_stat")

# Clustering using kmeans algorithm
kmeans_model <- kmeans(data_transformed, centers = 2)

kmeans_model

fviz_cluster(kmeans_model, data = data_transformed, palette = c("#2E9FDF", "#E7B800", "#00AFBB",  "#b800e7"),
             geom = "point",
             ellipse.type = "convex",
             ggtheme = theme_bw())


wss <- kmeans_model$tot.withinss
bss <- kmeans_model$betweenss
tss <- wss + bss
bss_ratio <- bss / tss

wss
bss
tss
bss_ratio

# Silhouette
sil <- silhouette(kmeans_model$cluster, dist(data_transformed))
fviz_silhouette(sil)

install.packages("fpc")
library(fpc) 

# Obtain the cluster assignments
clustering <- kmeans_model$cluster

# Compute the Calinski-Harabasz Index
calinski_harabasz <- calinhara(data_transformed, clustering)

# Print the Calinski-Harabasz Index
print(calinski_harabasz)


# Compute Calinski-Harabasz Index for kmeans clustering with varying number of clusters
calinski_results <- numeric(10)
for (k in 2:10) {
  kmeans_model <- kmeans(data_transformed, centers = k)
  clustering <- kmeans_model$cluster
  calinski_results[k] <- calinhara(data_transformed, clustering)
}

# Visualize the index
plot(2:10, calinski_results[2:10], type = "b", xlab = "Number of Clusters", ylab = "Calinski-Harabasz Index")
