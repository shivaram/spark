package spark

/**
 * Base class for dependencies.
 */
abstract class Dependency[T](val rdd: RDD[T]) extends Serializable

/**
 * Base class for dependencies where each partition of the parent RDD is used by at most one
 * partition of the child RDD.  Narrow dependencies allow for pipelined execution.
 */
abstract class NarrowDependency[T](rdd: RDD[T]) extends Dependency(rdd) {
  /**
   * Get the parent partitions for a child partition.
   * @param outputPartition a partition of the child RDD
   * @return the partitions of the parent RDD that the child partition depends upon
   */
  def getParents(outputPartition: Int): Seq[Int]
}

/**
 * Represents a dependency on the output of a shuffle stage.
 * @param shuffleId the shuffle id
 * @param rdd the parent RDD
 * @param aggregator optional aggregator; this allows for map-side combining
 * @param partitioner partitioner used to partition the shuffle output
 */
class ShuffleDependency[K, V, C](
    val shuffleId: Int,
    @transient rdd: RDD[(K, V)],
    val aggregator: Option[Aggregator[K, V, C]],
    val partitioner: Partitioner)
  extends Dependency(rdd)

/**
 * Represents a one-to-one dependency between partitions of the parent and child RDDs.
 */
class OneToOneDependency[T](rdd: RDD[T]) extends NarrowDependency[T](rdd) {
  override def getParents(partitionId: Int) = List(partitionId)
}

/**
 * Represents a one-to-one dependency between ranges of partitions in the parent and child RDDs.
 * @param rdd the parent RDD
 * @param inStart the start of the range in the parent RDD
 * @param outStart the start of the range in the child RDD
 * @param length the length of the range
 */
class RangeDependency[T](rdd: RDD[T], inStart: Int, outStart: Int, length: Int)
  extends NarrowDependency[T](rdd) {
  
  override def getParents(partitionId: Int) = {
    if (partitionId >= outStart && partitionId < outStart + length) {
      List(partitionId - outStart + inStart)
    } else {
      Nil
    }
  }
}
