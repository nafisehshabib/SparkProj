/**
  * Created by Nafiseh on 11/12/15.
  */


import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

class KMeans {
  val conf = new SparkConf() //
    .set("spark.driver.maxResultSize", "3g")
    //.set("spark.executor.memory", "6g")
    .setAppName("walmart")
    .setMaster("local[2]")
    .set("spark.driver.cores", "3")//.set(

  val sc = new SparkContext(conf)


  // Load and parse the data
  val data = sc.textFile("src/main/resources/kmeans_data.txt")
  val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

  // Cluster the data into two classes using KMeans
  val numClusters = 2
  val numIterations = 20
  val clusters = KMeans.train(parsedData, numClusters, numIterations)

  // Evaluate clustering by computing Within Set Sum of Squared Errors
  val WSSSE = clusters.computeCost(parsedData)
  println("Within Set Sum of Squared Errors = " + WSSSE)

  // Save and load model
  clusters.save(sc, "src/main/resources/kMeansRes")



}
