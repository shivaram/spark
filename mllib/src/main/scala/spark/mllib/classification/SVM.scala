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

package spark.mllib.classification

import scala.math.signum
import spark.{Logging, RDD, SparkContext}
import spark.mllib.optimization._
import spark.mllib.regression._
import spark.mllib.util.MLUtils

import org.jblas.DoubleMatrix

/**
 * SVM using Stochastic Gradient Descent.
 */
class SVMModel(
    override val weights: Array[Double],
    override val intercept: Double)
  extends GeneralizedLinearModel[Int](weights, intercept)
  with ClassificationModel with Serializable {

  override def predictPoint(dataMatrix: DoubleMatrix, weightMatrix: DoubleMatrix,
      intercept: Double) = {
    signum(dataMatrix.dot(weightMatrix) + intercept).toInt
  }
}

class SVMWithSGD private (
    var stepSize: Double,
    var numIterations: Int,
    var regParam: Double,
    var miniBatchFraction: Double,
    var addIntercept: Boolean)
  extends GeneralizedLinearAlgorithm[Int, SVMModel] with GradientDescent with Serializable {

  val gradient = new HingeGradient()
  val updater = new SquaredL2Updater()

  /**
   * Construct a SVM object with default parameters
   */
  def this() = this(1.0, 100, 1.0, 1.0, true)

  def createModel(weights: Array[Double], intercept: Double) = {
    new SVMModel(weights, intercept)
  }
}

/**
 * Top-level methods for calling SVM.
 */
object SVMWithSGD extends GLMWithSGD[Int, SVMModel] {

  val glmAlgorithm = new SVMWithSGD()

  def main(args: Array[String]) {
    if (args.length != 5) {
      println("Usage: SVM <master> <input_dir> <step_size> <regularization_parameter> <niters>")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "SVM")
    val data = MLUtils.loadLabeledData(sc, args(1)).map(yx => (yx._1.toInt, yx._2))
    val model = SVMWithSGD.trainSGD(data, args(4).toInt, args(2).toDouble, args(3).toDouble)

    sc.stop()
  }
}
