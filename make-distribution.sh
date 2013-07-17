#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Script to create a binary distribution for easy deploys of Spark.
# The distribution directory defaults to dist/ but can be overridden below.
# The distribution contains fat (assembly) jars that include the Scala library,
# so it is completely self contained.
# It does not contain source or *.class files.
#
# Arguments
#   (none): Creates dist/ directory
#      tgz: Additionally creates spark-$VERSION-bin.tar.gz
#
# Recommended deploy/testing procedure (standalone mode):
# 1) Rsync / deploy the dist/ dir to one host
# 2) cd to deploy dir; ./bin/start-master.sh
# 3) Verify master is up by visiting web page, ie http://master-ip:8080.  Note the spark:// URL.
# 4) ./bin/start-slave.sh 1 <<spark:// URL>>
# 5) MASTER="spark://my-master-ip:7077" ./spark-shell
#

# Figure out where the Spark framework is installed
FWDIR="$(cd `dirname $0`; pwd)"
DISTDIR="$FWDIR/dist"

# Get version from SBT
export TERM=dumb   # Prevents color codes in SBT output
VERSION=$($FWDIR/sbt/sbt "show version" | tail -1 | cut -f 2 | sed 's/^\([a-zA-Z0-9.-]*\).*/\1/')

if [ "$1" == "tgz" ]; then
	echo "Making spark-$VERSION-bin.tar.gz"
else
	echo "Making distribution for Spark $VERSION in $DISTDIR..."
fi


# Build fat JAR
$FWDIR/sbt/sbt "repl/assembly"

# Make directories
rm -rf "$DISTDIR"
mkdir -p "$DISTDIR/jars"
echo "$VERSION" >$DISTDIR/RELEASE

# Copy jars
cp $FWDIR/repl/target/*.jar "$DISTDIR/jars/"

# Copy other things
cp -r "$FWDIR/bin" "$DISTDIR"
cp -r "$FWDIR/conf" "$DISTDIR"
cp "$FWDIR/run" "$FWDIR/spark-shell" "$DISTDIR"


if [ "$1" == "tgz" ]; then
  TARDIR="$FWDIR/spark-$VERSION"
  cp -r $DISTDIR $TARDIR
  tar -zcf spark-$VERSION-bin.tar.gz -C $FWDIR spark-$VERSION
  rm -rf $TARDIR
fi
