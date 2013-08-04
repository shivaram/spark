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

package spark.mllib.regression

import spark.{Logging, RDD, SparkContext}
import spark.mllib.optimization._
import spark.mllib.util.MLUtils

import org.jblas.DoubleMatrix

/**
 * Lasso using Stochastic Gradient Descent.
 *
 */
class LassoModel(
    override val weights: Array[Double],
    override val intercept: Double)
  extends GeneralizedLinearModel[Double](weights, intercept)
  with RegressionModel with Serializable {

  override def predictPoint(dataMatrix: DoubleMatrix, weightMatrix: DoubleMatrix,
      intercept: Double) = {
    dataMatrix.dot(weightMatrix) + intercept
  }
}


class LassoWithSGD (
    var stepSize: Double,
    var numIterations: Int,
    var regParam: Double,
    var miniBatchFraction: Double,
    var addIntercept: Boolean)
  extends GeneralizedLinearAlgorithm[Double, LassoModel]
  with GradientDescent with Serializable {

  val gradient = new SquaredGradient()
  val updater = new L1Updater()

  /**
   * Construct a Lasso object with default parameters
   */
  def this() = this(1.0, 100, 1.0, 1.0, true)

  def createModel(weights: Array[Double], intercept: Double) = {
    new LassoModel(weights, intercept)
  }
}

/**
 * Top-level methods for calling Lasso.
 */
object LassoWithSGD extends GLMWithSGD[Double, LassoModel, LassoWithSGD](new LassoWithSGD()) {

  def main(args: Array[String]) {
    if (args.length != 5) {
      println("Usage: Lasso <master> <input_dir> <step_size> <regularization_parameter> <niters>")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "Lasso")
    val data = MLUtils.loadLabeledData(sc, args(1))
    val model = LassoWithSGD.trainSGD(data, args(4).toInt, args(2).toDouble, args(3).toDouble)

    sc.stop()
  }
}
