package spark

abstract class OutputFunction[T, U] extends Serializable {
  var context:TaskContext = null
  def setContext(ctx: TaskContext) {
    context = ctx
  }
  def initializer() : Unit = {}
  def finalizer() : Unit = {}
  def process(element: T)
  def output(): U
}
