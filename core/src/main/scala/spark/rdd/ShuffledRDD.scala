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

package spark.rdd

import spark.{Partitioner, RDD, SparkEnv, ShuffleDependency, Partition, TaskContext}
import spark.SparkContext._


private[spark] class ShuffledRDDPartition(val idx: Int) extends Partition {
  override val index = idx
  override def hashCode(): Int = idx
}

/**
 * The resulting RDD from a shuffle (e.g. repartitioning of data).
 * @param prev the parent RDD.
 * @param part the partitioner used to partition the RDD
 * @param serializerClass class name of the serializer to use.
 * @tparam K the key class.
 * @tparam V the value class.
 */
class ShuffledRDD[K, V](
    @transient prev: RDD[(K, V)],
    part: Partitioner,
    serializerClass: String = null)
  extends RDD[(K, V)](prev.context, List(new ShuffleDependency(prev, part, serializerClass))) {

  override val partitioner = Some(part)

  override def getPartitions: Array[Partition] = {
    Array.tabulate[Partition](part.numPartitions)(i => new ShuffledRDDPartition(i))
  }

  override def compute(split: Partition, context: TaskContext): Iterator[(K, V)] = {
    val shuffledId = dependencies.head.asInstanceOf[ShuffleDependency[K, V]].shuffleId
    SparkEnv.get.shuffleFetcher.fetch[K, V](shuffledId, split.index, context.taskMetrics,
      SparkEnv.get.serializerManager.get(serializerClass))
  }
}
