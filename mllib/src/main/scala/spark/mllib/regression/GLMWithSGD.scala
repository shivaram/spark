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

import spark.RDD
import spark.mllib.optimization._

/**
 * Helper class that generates top-level methods that can be used to
 * call GLMs easily.
 */
class GLMWithSGD[T, M, A] (glmAlgorithm: A) (implicit
  t: T => Double,
  methodEv: M <:< GeneralizedLinearModel[T],
  mt: Manifest[M],
  algEv: A <:< GeneralizedLinearAlgorithm[T, M] with GradientDescent) {

  /**
   * Train a GLM model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
   * gradient descent are initialized using the initial weights provided.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param regParam Regularization parameter.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   * @param initialWeights Initial set of weights to be used. Array should be equal in size to 
   *        the number of features in the data.
   */
  def trainSGD(
      input: RDD[(T, Array[Double])],
      numIterations: Int,
      stepSize: Double,
      regParam: Double,
      miniBatchFraction: Double,
      initialWeights: Array[Double])
    : M =
  {
    glmAlgorithm.setStepSize(stepSize)
                .setNumIterations(numIterations)
                .setRegParam(regParam)
                .setMiniBatchFraction(miniBatchFraction)
                .setIntercept(true)
                .train(input, initialWeights)
  }

  /**
   * Train a GLM model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param regParam Regularization parameter.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   */
  def trainSGD(
      input: RDD[(T, Array[Double])],
      numIterations: Int,
      stepSize: Double,
      regParam: Double,
      miniBatchFraction: Double)
    : M =
  {

    glmAlgorithm.setStepSize(stepSize)
                .setNumIterations(numIterations)
                .setRegParam(regParam)
                .setMiniBatchFraction(miniBatchFraction)
                .setIntercept(true)
                .train(input)
  }

  /**
   * Train a GLM model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. We use the entire data set to
   * update the gradient in each iteration.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param stepSize Step size to be used for each iteration of Gradient Descent.
   * @param regParam Regularization parameter.
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a GLM which has the weights and offset from training.
   */
  def trainSGD(
      input: RDD[(T, Array[Double])],
      numIterations: Int,
      stepSize: Double,
      regParam: Double)
    : M =
  {
    trainSGD(input, numIterations, stepSize, regParam, 1.0)
  }

  /**
   * Train a GLM model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using a step size of 1.0. We use the entire data set to
   * update the gradient in each iteration.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a GLM which has the weights and offset from training.
   */
  def trainSGD(
      input: RDD[(T, Array[Double])],
      numIterations: Int)
    : M =
  {
    trainSGD(input, numIterations, 1.0, 1.0, 1.0)
  }

}
