package spark.examples

import java.util.Date
import java.util.Random
import scala.actors._
import scala.actors.Actor._
import scala.math.sqrt
import cern.jet.math._
import cern.colt.matrix._
import cern.colt.matrix.linalg._

// Every update message is a 1D update and an index which is updated
case class Matrix1DUpdate(update: DoubleMatrix1D, index: Int, 
                          client: Actor)
case class Iterate(index: Int)

trait Updater {
  val LAMBDA = 0.01 // Regularization coefficient

  // Some COLT objects
  val factory2D = DoubleFactory2D.dense
  val factory1D = DoubleFactory1D.dense
  val algebra = Algebra.DEFAULT
  val blas = SeqBlas.seqBlas


  def rmse(targetR: DoubleMatrix2D, ms: Array[DoubleMatrix1D],
    us: Array[DoubleMatrix1D], M: Int, U: Int, F: Int): Double =
  {
    val r = factory2D.make(M, U)
    for (i <- 0 until M; j <- 0 until U) {
      r.set(i, j, blas.ddot(ms(i), us(j)))
    }
    blas.daxpy(-1, targetR, r)
    val sumSqs = r.aggregate(Functions.plus, Functions.square)
    return sqrt(sumSqs / (M * U))
  }
}

class MovieUpdater(us: Array[DoubleMatrix1D], userActor: Actor,
    R: DoubleMatrix2D, ITERATIONS: Int, M: Int, F: Int, U: Int) extends Actor with Updater {
  var numIterations: Int = 0;
  var ms = Array.fill(M)(factory1D.make(F))
  var startT = new Date().getTime()
  var endT = new Date().getTime()
  var veryBeginning = new Date().getTime()
  var nextIndex: Int = 0

  def act() {
    self ! Iterate(0)
    loop {
      react {
        case Matrix1DUpdate(update, index, client) =>
          us(index) = update

        case Iterate(index) =>
          val XtX = factory2D.make(F, F)
          val Xty = factory1D.make(F)
          // For each user that rated the movie
          for (j <- 0 until U) {
            val u = us(j)
            // Add u * u^t to XtX
            blas.dger(1, u, u, XtX)
            // Add u * rating to Xty
            blas.daxpy(R.get(index, j), u, Xty)
          }
          // Add regularization coefs to diagonal terms
          for (d <- 0 until F) {
            XtX.set(d, d, XtX.get(d, d) + LAMBDA * U)
          }
          // Solve it with Cholesky
          val ch = new CholeskyDecomposition(XtX)
          val Xty2D = factory2D.make(Xty.toArray, F)
          val solved2D = ch.solve(Xty2D)
          ms(index) = solved2D.viewColumn(0)

          // send update to user actor
          userActor ! Matrix1DUpdate(ms(index), index, self)
          nextIndex = index + 1
          if (nextIndex == M) {
            endT = new Date().getTime()
            nextIndex = 0
            numIterations = numIterations + 1
            println("Movies: Iteration " + numIterations + 
                " RMSE = " + rmse(R, ms, us, M, U, F) + " time " + 
                (endT - startT) + " ms")
            startT = endT
            if (numIterations == ITERATIONS) {
              var veryEnd = new Date().getTime()
              println("Movies total time taken: " + 
                      (veryEnd - veryBeginning) + " ms");
              exit()
            }
          }
          // schedule the next iteration
          self ! Iterate(nextIndex)
      }
    }
  }
}

class UserUpdater(ms: Array[DoubleMatrix1D], R: DoubleMatrix2D, 
    ITERATIONS: Int, M: Int, F: Int, U: Int) extends Actor with Updater {
  var numIterations: Int = 0;
  var us = Array.fill(U)(factory1D.make(F))
  var startT = new Date().getTime()
  var endT = new Date().getTime()

  def act() {
    var movieActor: Actor = null
    loop {
      var nextIndex: Int = 0
      react {
        case Matrix1DUpdate(update, index, client) =>
          ms(index) = update
          if (movieActor == null) {
            movieActor = client
            self ! Iterate(0)
          }

        case Iterate(index) =>
          val XtX = factory2D.make(F, F)
          val Xty = factory1D.make(F)
          // For each movie that the user rated
          for (i <- 0 until M) {
            val m = ms(i)
            // Add m * m^t to XtX
            blas.dger(1, m, m, XtX)
            // Add m * rating to Xty
            blas.daxpy(R.get(i, index), m, Xty)
          }
          // Add regularization coefs to diagonal terms
          for (d <- 0 until F) {
            XtX.set(d, d, XtX.get(d, d) + LAMBDA * M)
          }
          // Solve it with Cholesky
          val ch = new CholeskyDecomposition(XtX)
          val Xty2D = factory2D.make(Xty.toArray, F)
          val solved2D = ch.solve(Xty2D)
          us(index) = solved2D.viewColumn(0)

          // Send the update to movie actor
          movieActor ! Matrix1DUpdate(us(index), index, self)
          nextIndex = index + 1
          if (nextIndex == U) {
            endT = new Date().getTime()
            nextIndex = 0
            numIterations = numIterations + 1
            print("Users: Iteration " + numIterations + 
                " RMSE = " + rmse(R, ms, us, M, U, F) + " time " +
                (endT - startT) + " ms")
            println()
            startT = endT
            if (numIterations == ITERATIONS) {
              exit()
            }
          }
          // schedule the next iteration
          self ! Iterate(nextIndex)
      }
    }
  }
}

object LocalALSActors {
  // Parameters set through command line arguments
  var M = 0 // Number of movies
  var U = 0 // Number of users
  var F = 0 // Number of features
  var ITERATIONS = 0

  // Some COLT objects
  val factory2D = DoubleFactory2D.dense
  val factory1D = DoubleFactory1D.dense
  val algebra = Algebra.DEFAULT
  val blas = SeqBlas.seqBlas

  def generateR(): DoubleMatrix2D = {
    val mh = factory2D.random(M, F)
    val uh = factory2D.random(U, F)
    return algebra.mult(mh, algebra.transpose(uh))
  }

  def main(args: Array[String]) {
    args match {
      case Array(m, u, f, iters) => {
        M = m.toInt
        U = u.toInt
        F = f.toInt
        ITERATIONS = iters.toInt
      }
      case _ => {
        System.err.println("Usage: LocalALS <M> <U> <F> <iters>")
        System.exit(1)
      }
    }
    printf("Running with M=%d, U=%d, F=%d, iters=%d\n", M, U, F, ITERATIONS);
    
    val R = generateR()

    // Initialize m and u randomly
    var ms = Array.fill(M)(factory1D.random(F))
    var us = Array.fill(U)(factory1D.random(F))

    val uU = new UserUpdater(ms, R, ITERATIONS, M, F, U)
    val mU = new MovieUpdater(us, uU, R, ITERATIONS, M, F, U)

    uU.start
    mU.start
  }
}
