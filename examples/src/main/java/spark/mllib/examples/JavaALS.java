/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package spark.mllib.examples;

import spark.api.java.JavaRDD;
import spark.api.java.JavaSparkContext;
import spark.api.java.function.Function;

import spark.mllib.recommendation.ALS;
import spark.mllib.recommendation.MatrixFactorizationModel;

import java.io.Serializable;
import java.util.Arrays;
import java.util.StringTokenizer;

import scala.Tuple3;
import scala.Tuple2;

/**
 * Example using MLLib ALS from Java.
 */
public class  JavaALS {

  static class ParseRating extends Function<String, Tuple3<Integer, Integer, Double>> {
    public Tuple3<Integer, Integer, Double> call(String line) {
      StringTokenizer tok = new StringTokenizer(line, ",");
      Integer x = Integer.parseInt(tok.nextToken());
      Integer y = Integer.parseInt(tok.nextToken());
      Double rating = Double.parseDouble(tok.nextToken());
      return new Tuple3(x, y, rating);
    }
  }

  static class FeaturesToString extends Function<Tuple2<Object, double[]>, String> {
    public String call(Tuple2<Object, double[]> element) {
      return element._1().toString() + "," + Arrays.toString(element._2());
    }
  }

  public static void main(String[] args) {

    if (args.length != 5 && args.length != 6) {
      System.err.println(
          "Usage: JavaALS <master> <ratings_file> <rank> <iterations> <output_dir> [<blocks>]");
      System.exit(1);
    }

    int rank = Integer.parseInt(args[2]);
    int iterations = Integer.parseInt(args[3]);
    String outputDir = args[4];
    int blocks = -1;
    if (args.length == 6) {
      blocks = Integer.parseInt(args[5]);
    }

    JavaSparkContext sc = new JavaSparkContext(args[0], "JavaALS",
        System.getenv("SPARK_HOME"), System.getenv("SPARK_EXAMPLES_JAR"));
    JavaRDD<String> lines = sc.textFile(args[1]);

    JavaRDD<Tuple3<Integer, Integer, Double>> ratings = lines.map(new ParseRating());

    MatrixFactorizationModel model = ALS.trainALS(ratings.rdd(), rank, iterations, 0.01, blocks);

    // TODO(shivaram): Uncomment this after we have a method to convert
    // scala RDDs to JavaRDDs
    // model.userFeatures().toJavaRDD().map(new FeaturesToString()).saveAsTextFile(
    //     outputDir + "/userFeatures");
    // model.productFeatures().toJavaRDD().map(new FeaturesToString()).saveAsTextFile(
    //     outputDir + "/productFeatures");
    System.out.println("Final user/product features written to " + outputDir);

    System.exit(0);
  }
}
