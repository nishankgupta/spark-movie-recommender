package com.nishank.spark

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Recommender extends App {

  main

  def main(): Unit = {
    val spark = SparkSession.builder.appName("Spark-Reccommender").master("local").getOrCreate()

    //loading ratings into DF
    val ratingsDF = spark.read.format("com.databricks.spark.csv").option("header", true)
      .load("src/main/resources/movielens/ratings.csv")
    println(ratingsDF.schema.treeString)


    //loading movies into DF
    val moviesDF = spark.read.format("com.databricks.spark.csv").option("header", true)
      .load("src/main/resources/movielens/ratings.csv")
    println(moviesDF.schema.treeString)

    ratingsDF.createOrReplaceTempView("ratings")
    moviesDF.createOrReplaceTempView("movies")

    val numRatings = ratingsDF.count;
    val numUsers = ratingsDF.select(ratingsDF.col("userId")).distinct().count();
    val numMovies = ratingsDF.select(ratingsDF.col("movieId")).distinct().count()
    println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.")

    //showStats(spark)

    //Split ratings RDD into training RDD (75%) & test RDD (25%)
    val splits = ratingsDF.randomSplit(Array(0.72, 0.25), 12345L)

    val (trainingData, testData) = (splits(0), splits(1))

    val numTraining = trainingData.count()
    val numTest = testData.count()
    println("Training: " + numTraining + " test: " + numTest)

    val ratingsRDD = getDataRdd(trainingData)
    val testRDD = getDataRdd(testData)

    //prepare model
    val model = getALPModel(ratingsRDD)

    doPrediction(model, 414, 5)
  }

  def getALPModel(ratingsRDD: RDD[Rating]) = {
    val rank = 20
    val numIterations = 15
    val lambda = 0.10
    val alpha = 1.00
    val block = -1
    val seed = 12345L
    val implicitPrefs = false

    new ALS().setIterations(numIterations).setBlocks(block).setAlpha(alpha).setLambda(lambda)
      .setRank(rank).setSeed(seed)
      .setImplicitPrefs(implicitPrefs)
      .run(ratingsRDD)
  }

  def doPrediction(model: MatrixFactorizationModel, userId: Int, numRecommendation: Int) = {
    val sb = new StringBuilder
    sb.append("Rating:(UserID, MovieID, Rating)").append("\n")
    sb.append("----------------------------------").append("\n")

    val topRecsForUser = model.recommendProducts(userId, numRecommendation)

    for (rating <- topRecsForUser) {
      sb.append(rating.toString()).append("\n")
      sb.append("----------------------------------").append("\n")
    }

    println(sb.toString())
  }

  def getDataRdd(dataset: Dataset[Row]) = {
    dataset.rdd.map(row => {
      val userId = row.getString(0)

      val movieId = row.getString(1)

      val ratings = row.getString(2)

      Rating(userId.toInt, movieId.toInt, ratings.toDouble)
    })
  }


  def showStats(spark: SparkSession): Unit = {
    // Get the max, min ratings along with the count of users who have rated a movie.
    val movieRatingsDF = spark.sql("select movies.title, movierates.maxr, movierates.minr, movierates.cntu "

      + "from(SELECT ratings.movieId,max(ratings.rating) as maxr,"

      + "min(ratings.rating) as minr,count(distinct userId) as cntu "

      + "FROM ratings group by ratings.movieId) movierates "

      + "join movies on movierates.movieId=movies.movieId "

      + "order by movierates.cntu desc")

    movieRatingsDF.show(false)


    //10 most active users
    val mostActiveUsersSchemaRDD = spark.sql("SELECT ratings.userId, count(*) as ct from ratings "
      + "group by ratings.userId order by ct desc limit 10")
    mostActiveUsersSchemaRDD.show(false)

    //deatils about user 414
    val specificUserDF = spark.sql(

      "SELECT ratings.userId, ratings.movieId,"

        + "ratings.rating, movies.title FROM ratings JOIN movies "

        + "ON movies.movieId=ratings.movieId "

        + "where ratings.userId=414 and ratings.rating > 4")
    specificUserDF.show(false)
  }


}
