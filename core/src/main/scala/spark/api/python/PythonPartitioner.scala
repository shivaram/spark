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

package spark.api.python

import spark.Partitioner

import java.util.Arrays

/**
 * A [[spark.Partitioner]] that performs handling of byte arrays, for use by the Python API.
 *
 * Stores the unique id() of the Python-side partitioning function so that it is incorporated into
 * equality comparisons.  Correctness requires that the id is a unique identifier for the
 * lifetime of the program (i.e. that it is not re-used as the id of a different partitioning
 * function).  This can be ensured by using the Python id() function and maintaining a reference
 * to the Python partitioning function so that its id() is not reused.
 */
private[spark] class PythonPartitioner(
  override val numPartitions: Int,
  val pyPartitionFunctionId: Long)
  extends Partitioner {

  override def getPartition(key: Any): Int = {
    if (key == null) {
      return 0
    }
    else {
      val hashCode = {
        if (key.isInstanceOf[Array[Byte]]) {
          Arrays.hashCode(key.asInstanceOf[Array[Byte]])
        } else {
          key.hashCode()
        }
      }
      val mod = hashCode % numPartitions
      if (mod < 0) {
        mod + numPartitions
      } else {
        mod // Guard against negative hash codes
      }
    }
  }

  override def equals(other: Any): Boolean = other match {
    case h: PythonPartitioner =>
      h.numPartitions == numPartitions && h.pyPartitionFunctionId == pyPartitionFunctionId
    case _ =>
      false
  }
}
