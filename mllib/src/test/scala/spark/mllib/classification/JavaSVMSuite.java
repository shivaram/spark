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

package spark.mllib.classification;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

import scala.Tuple2;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import spark.api.java.JavaDoubleRDD;
import spark.api.java.JavaPairRDD;
import spark.api.java.JavaRDD;
import spark.api.java.JavaSparkContext;
import spark.api.java.function.*;
import scala.collection.Seq;
import scala.collection.*;


public class JavaSVMSuite implements Serializable {
  private transient JavaSparkContext sc;

  @Before
  public void setUp() {
    sc = new JavaSparkContext("local", "JavaSVMSuite");
  }

  @After
  public void tearDown() {
    sc.stop();
    sc = null;
    System.clearProperty("spark.driver.port");
  }

  @Test
  public void runSVMUsingConstructor() {
    int nPoints = 10000;
    double A = 2.0;
    double[] weights = {-1.5, 1.0};

    JavaRDD<Tuple2<Object, double[]>> testRDD = sc.parallelize(SVMSuite.generateSVMInputAsList(A,
        weights, nPoints, 42), 2).cache();

    SVMWithSGD svmSGDImpl = new SVMWithSGD();
    svmSGDImpl.setStepSize(1.0)
              .setRegParam(1.0)
              .setNumIterations(100);

    SVMModel model = svmSGDImpl.train(testRDD.rdd());

    List<Tuple2<Object, double[]>> validationData =
        SVMSuite.generateSVMInputAsList(A, weights, nPoints, 17);

    int numAccurate = 0;
    for (Tuple2<Object, double[]> point: validationData) {
      Integer prediction = (Integer) model.predict(point._2());
      if ((Integer)prediction == point._1()) {
        numAccurate++;
      }
    }

    Assert.assertTrue(numAccurate > nPoints * 4.0 / 5.0);
  }

  @Test
  public void runSVMUsingStaticMethods() {
    int nPoints = 10000;
    double A = 2.0;
    double[] weights = {-1.5, 1.0};

    JavaRDD<Tuple2<Object, double[]>> testRDD = sc.parallelize(SVMSuite.generateSVMInputAsList(A,
        weights, nPoints, 42), 2).cache();

    SVMModel model = SVMWithSGD.trainSGD(testRDD.rdd(), 100, 1.0, 1.0, 1.0);

    List<Tuple2<Object, double[]>> validationData =
        SVMSuite.generateSVMInputAsList(A, weights, nPoints, 17);

    int numAccurate = 0;
    for (Tuple2<Object, double[]> point: validationData) {
      Integer prediction = (Integer) model.predict(point._2());
      if ((Integer)prediction == point._1()) {
        numAccurate++;
      }
    }

    Assert.assertTrue(numAccurate > nPoints * 4.0 / 5.0);
  }

}
