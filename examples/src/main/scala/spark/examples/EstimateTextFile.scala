package spark.examples

import javax.management.MBeanServer
import java.lang.management.ManagementFactory
import com.sun.management.HotSpotDiagnosticMXBean

import spark._

object EstimateTextFile {
  // Parameters set through command line arguments
  var fileName: String = _
  var hostName: String = _

  def main(args: Array[String]) {
    args match {
      case Array(host, file) => {
        hostName = host
        fileName = file
      }
      case _ => {
        System.err.println("Usage: EstimateTextFile <hostname> <filename>")
        System.exit(1)
      }
    }
    printf("Running with hostname %s, filename %s\n", hostName, fileName);
    
    var sc = new SparkContext(hostName, "EstimateTextFile")

    val fileRDD = sc.textFile(fileName).cache
    val fileCount = fileRDD.count
    println("Number of lines in file " + fileCount)

    val hotSpotMBeanName = "com.sun.management:type=HotSpotDiagnostic";
    val server = ManagementFactory.getPlatformMBeanServer();
    val bean = ManagementFactory.newPlatformMXBeanProxy(server, 
      hotSpotMBeanName, classOf[HotSpotDiagnosticMXBean]);
    bean.dumpHeap("heap.bin", true)
  }
}
