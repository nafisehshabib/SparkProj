import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by Nafiseh on 11/12/15.
  */
object start {


  def main(args: Array[String]) {
    val conf = new SparkConf() //
      .setMaster("local[1]")
      .setAppName("KMean-test")
      .set("spark.driver.cores", "3")

    val sc = new SparkContext(conf)
    val kmean = new KMeans(sc)
  }
}
