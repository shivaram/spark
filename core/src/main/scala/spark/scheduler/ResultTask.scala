package spark.scheduler

import spark._

class ResultTask[T, U](
    stageId: Int,
    rdd: RDD[T],
    func: OutputFunction[T, U],
    val partition: Int,
    @transient locs: Seq[String],
    val outputId: Int)
  extends Task[U](stageId) {
  
  val split = rdd.splits(partition)

  override def run(attemptId: Long): U = {
    val context = new TaskContext(stageId, partition, attemptId)
    func.setContext(context)
    func.initializer()
    for (elem <- rdd.iterator(split)) {
      // TODO: Check if thread is interrupted here
      func.process(elem)
    }
    func.finalizer()
    func.output()
  }

  override def preferredLocations: Seq[String] = locs

  override def toString = "ResultTask(" + stageId + ", " + partition + ")"
}
