package spark.ui

import scala.xml.Node

import spark.SparkContext

/** Utility functions for generating XML pages with spark content. */
private[spark] object UIUtils {
  import Page._

  /** Returns a spark page with correctly formatted headers */
  def headerSparkPage(content: => Seq[Node], sc: SparkContext, title: String, page: Page.Value)
  : Seq[Node] = {
    val storage = page match {
      case Storage => <li class="active"><a href="/storage">Storage</a></li>
      case _ => <li><a href="/storage">Storage</a></li>
    }
    val jobs = page match {
      case Jobs => <li class="active"><a href="/stages">Jobs</a></li>
      case _ => <li><a href="/stages">Jobs</a></li>
    }

    <html>
      <head>
        <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
        <link rel="stylesheet" href="/static/bootstrap.min.css" type="text/css" />
        <link rel="stylesheet" href="/static/webui.css" type="text/css" />
        <link rel="stylesheet" href="/static/bootstrap-responsive.min.css" type="text/css" />
        <script src="/static/sorttable.js"></script>
        <title>{title}</title>
        <style type="text/css">
          table.sortable thead {{ cursor: pointer; }}
        </style>
      </head>
      <body>
        <div class="container">

          <div class="row">
            <div class="span12">
              <div class="navbar">
                <div class="navbar-inner">
                  <div class="container">
                    <div class="brand"><img src="/static/spark-logo-77x50px-hd.png" /></div>
                    <ul class="nav">
                      {storage}
                      {jobs}
                    </ul>
                    <ul id="infolist">
                      <li>Application: <strong>{sc.appName}</strong></li>
                      <li>Master: <strong>{sc.master}</strong></li>
                      <li>Executors: <strong>{sc.getExecutorStorageStatus.size}</strong></li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="row" style="padding-top: 5px;">
            <div class="span12">
              <h1 style="vertical-align: bottom; display: inline-block;">
                {title}
              </h1>
            </div>
          </div>
          <hr/>
          {content}
        </div>
      </body>
    </html>
  }

  /** Returns a page with the spark css/js and a simple format. Used for scheduler UI. */
  def basicSparkPage(content: => Seq[Node], title: String): Seq[Node] = {
    <html>
      <head>
        <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
        <link rel="stylesheet" href="/static/bootstrap.min.css" type="text/css" />
        <link rel="stylesheet" href="/static/bootstrap-responsive.min.css" type="text/css" />
        <script src="/static/sorttable.js"></script>
        <title>{title}</title>
        <style type="text/css">
          table.sortable thead {{ cursor: pointer; }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="row">
            <div class="span2">
              <img src="/static/spark_logo.png" />
            </div>
            <div class="span10">
              <h1 style="vertical-align: bottom; margin-top: 40px; display: inline-block;">
                {title}
              </h1>
            </div>
          </div>
          {content}
        </div>
      </body>
    </html>
  }

  /** Returns an HTML table constructed by generating a row for each object in a sequence. */
  def listingTable[T](headers: Seq[String], makeRow: T => Seq[Node], rows: Seq[T]): Seq[Node] = {
    <table class="table table-bordered table-striped table-condensed sortable">
      <thead>{headers.map(h => <th>{h}</th>)}</thead>
      <tbody>
        {rows.map(r => makeRow(r))}
      </tbody>
    </table>
  }
}