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

package spark.ui.storage

import javax.servlet.http.HttpServletRequest

import scala.xml.Node

import spark.storage.{RDDInfo, StorageUtils}
import spark.Utils
import spark.ui.UIUtils._
import spark.ui.Page._

/** Page showing list of RDD's currently stored in the cluster */
private[spark] class IndexPage(parent: BlockManagerUI) {
  val sc = parent.sc

  def render(request: HttpServletRequest): Seq[Node] = {
    val storageStatusList = sc.getExecutorStorageStatus
    // Calculate macro-level statistics
    val maxMem = storageStatusList.map(_.maxMem).reduce(_+_)
    val remainingMem = storageStatusList.map(_.memRemaining).reduce(_+_)
    val diskSpaceUsed = storageStatusList.flatMap(_.blocks.values.map(_.diskSize))
      .reduceOption(_+_).getOrElse(0L)

    val rddHeaders = Seq(
      "RDD Name",
      "Storage Level",
      "Cached Partitions",
      "Fraction Partitions Cached",
      "Size in Memory",
      "Size on Disk")
    val rdds = StorageUtils.rddInfoFromStorageStatus(storageStatusList, sc)
    val rddTable = listingTable(rddHeaders, rddRow, rdds)

    val content =
      <div class="row">
        <div class="span12">
          <ul class="unstyled">
            <li><strong>Memory:</strong>
              {Utils.memoryBytesToString(maxMem - remainingMem)} Used
              ({Utils.memoryBytesToString(remainingMem)} Available) </li>
            <li><strong>Disk:</strong> {Utils.memoryBytesToString(diskSpaceUsed)} Used </li>
          </ul>
        </div>
      </div> ++ {rddTable};

    headerSparkPage(content, parent.sc, "Spark Storage ", Storage)
  }

  def rddRow(rdd: RDDInfo): Seq[Node] = {
    <tr>
      <td>
        <a href={"/storage/rdd?id=%s".format(rdd.id)}>
          {rdd.name}
        </a>
      </td>
      <td>{rdd.storageLevel.description}
      </td>
      <td>{rdd.numCachedPartitions}</td>
      <td>{rdd.numCachedPartitions / rdd.numPartitions.toDouble}</td>
      <td>{Utils.memoryBytesToString(rdd.memSize)}</td>
      <td>{Utils.memoryBytesToString(rdd.diskSize)}</td>
    </tr>
  }
}
