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

import spark.storage.{StorageStatus, StorageUtils}
import spark.ui.UIUtils._
import spark.Utils
import spark.storage.BlockManagerMasterActor.BlockStatus
import spark.ui.Page._

/** Page showing storage details for a given RDD */
private[spark] class RDDPage(parent: BlockManagerUI) {
  val sc = parent.sc

  def render(request: HttpServletRequest): Seq[Node] = {
    val id = request.getParameter("id")
    val prefix = "rdd_" + id.toString
    val storageStatusList = sc.getExecutorStorageStatus
    val filteredStorageStatusList = StorageUtils.
      filterStorageStatusByPrefix(storageStatusList, prefix)
    val rddInfo = StorageUtils.rddInfoFromStorageStatus(filteredStorageStatusList, sc).head

    val workerHeaders = Seq("Host", "Memory Usage", "Disk Usage")
    val workers = filteredStorageStatusList.map((prefix, _))
    val workerTable = listingTable(workerHeaders, workerRow, workers)

    val blockHeaders = Seq("Block Name", "Storage Level", "Size in Memory", "Size on Disk",
      "Locations")

    val blockStatuses = filteredStorageStatusList.flatMap(_.blocks).toArray.sortWith(_._1 < _._1)
    val blockLocations = StorageUtils.blockLocationsFromStorageStatus(filteredStorageStatusList)
    val blocks = blockStatuses.map {
      case(id, status) => (id, status, blockLocations.get(id).getOrElse(Seq("UNKNOWN")))
    }
    val blockTable = listingTable(blockHeaders, blockRow, blocks)

    val content =
      <div class="row">
        <div class="span12">
          <ul class="unstyled">
            <li>
              <strong>Storage Level:</strong>
              {rddInfo.storageLevel.description}
            </li>
            <li>
              <strong>Cached Partitions:</strong>
              {rddInfo.numCachedPartitions}
            </li>
            <li>
              <strong>Total Partitions:</strong>
              {rddInfo.numPartitions}
            </li>
            <li>
              <strong>Memory Size:</strong>
              {Utils.memoryBytesToString(rddInfo.memSize)}
            </li>
            <li>
              <strong>Disk Size:</strong>
              {Utils.memoryBytesToString(rddInfo.diskSize)}
            </li>
          </ul>
        </div>
      </div>
      <hr/>
      <div class="row">
        <div class="span12">
          {workerTable}
        </div>
      </div>
      <hr/>
      <div class="row">
        <div class="span12">
          <h3> RDD Summary </h3>
          {blockTable}
        </div>
      </div>;

    headerSparkPage(content, parent.sc, "RDD Info: " + rddInfo.name, Jobs)
  }

  def blockRow(row: (String, BlockStatus, Seq[String])): Seq[Node] = {
    val (id, block, locations) = row
    <tr>
      <td>{id}</td>
      <td>
        {block.storageLevel.description}
      </td>
      <td sorttable_customkey={block.memSize.toString}>
        {Utils.memoryBytesToString(block.memSize)}
      </td>
      <td sorttable_customkey={block.diskSize.toString}>
        {Utils.memoryBytesToString(block.diskSize)}
      </td>
      <td>
        {locations.map(l => <span>{l}<br/></span>)}
      </td>
    </tr>
  }

  def workerRow(worker: (String, StorageStatus)): Seq[Node] = {
    val (prefix, status) = worker
    <tr>
      <td>{status.blockManagerId.host + ":" + status.blockManagerId.port}</td>
      <td>
        {Utils.memoryBytesToString(status.memUsed(prefix))}
        ({Utils.memoryBytesToString(status.memRemaining)} Total Available)
      </td>
      <td>{Utils.memoryBytesToString(status.diskUsed(prefix))}</td>
    </tr>
  }
}
