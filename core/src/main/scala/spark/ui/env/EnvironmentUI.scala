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

package spark.ui.env

import javax.servlet.http.HttpServletRequest

import org.eclipse.jetty.server.Handler

import scala.collection.JavaConversions._
import scala.util.Properties

import spark.ui.JettyUtils._
import spark.ui.UIUtils.headerSparkPage
import spark.ui.Page.Environment
import spark.SparkContext
import spark.ui.UIUtils

import scala.xml.Node

private[spark] class EnvironmentUI(sc: SparkContext) {

  def getHandlers = Seq[(String, Handler)](
    ("/environment", (request: HttpServletRequest) => envDetails(request))
  )

  def envDetails(request: HttpServletRequest): Seq[Node] = {
    val jvmInformation = Seq(
      ("Java Version", "%s (%s)".format(Properties.javaVersion, Properties.javaVendor)),
      ("Java Home", Properties.javaHome),
      ("Scala Version", Properties.versionString),
      ("Scala Home", Properties.scalaHome)
    )
    def jvmRow(kv: (String, String)) = <tr><td>{kv._1}</td><td>{kv._2}</td></tr>
    def jvmTable = UIUtils.listingTable(Seq("Name", "Value"), jvmRow, jvmInformation)

    val properties = System.getProperties.iterator.toSeq
    val classPathProperty = properties
        .filter{case (k, v) => k.contains("java.class.path")}
        .headOption
        .getOrElse("", "")
    val sparkProperties = properties.filter(_._1.startsWith("spark"))
    val otherProperties = properties.diff(sparkProperties :+ classPathProperty)

    val propertyHeaders = Seq("Name", "Value")
    def propertyRow(kv: (String, String)) = <tr><td>{kv._1}</td><td>{kv._2}</td></tr>
    val sparkPropertyTable = UIUtils.listingTable(propertyHeaders, propertyRow, sparkProperties)
    val otherPropertyTable = UIUtils.listingTable(propertyHeaders, propertyRow, otherProperties)

    val classPathEntries = classPathProperty._2
        .split(System.getProperty("path.separator", ":"))
        .filterNot(e => e.isEmpty)
        .map(e => (e, "System Classpath"))
    val addedJars = sc.addedJars.iterator.toSeq.map{case (path, time) => (path, "Added By User")}
    val addedFiles = sc.addedFiles.iterator.toSeq.map{case (path, time) => (path, "Added By User")}
    val classPath = addedJars ++ addedFiles ++ classPathEntries

    val classPathHeaders = Seq("Resource", "Source")
    def classPathRow(data: (String, String)) = <tr><td>{data._1}</td><td>{data._2}</td></tr>
    val classPathTable = UIUtils.listingTable(classPathHeaders, classPathRow, classPath)

    val content =
      <span>
        <h2>Runtime Information</h2> {jvmTable}
        <h2>Spark Properties</h2> {sparkPropertyTable}
        <h2>System Properties</h2> {otherPropertyTable}
        <h2>Classpath Entries</h2> {classPathTable}
      </span>

    headerSparkPage(content, sc, "Environment", Environment)
  }
}
