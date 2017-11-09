/**
  * Created by Nafiseh on 11/12/15.
  */


import java.io.Serializable

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

class KMeans(sc: SparkContext) extends Serializable{


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


  val brc = sc.broadcast(clusters)
  val dataCluster = parsedData.map(data => (data, brc.value.predict(data)))
  dataCluster.foreach(dc => {
    val (d, c) = dc
    println(d+"  "+c)
  })

}
