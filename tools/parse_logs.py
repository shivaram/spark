#!/usr/bin/env python

from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from datetime import date
import re
import argparse

# Mapping
block_message_map = {}
message_block_map = defaultdict(list)

msg_begin_times = {}
msg_send_times = {}
msg_recv_times = {}
msg_ack_begin_times = {}
msg_ack_end_times = {}
msg_reply_times = {}

block_finish_times = {}
block_wall_times = {}

def get_datetime_from_line(line):
  parts = line.split(' ')
  date_str = parts[0] + " " + parts[1]
  return get_datetime_from_string(date_str)

def get_datetime_from_string(date_str):
  return datetime.strptime(date_str, '%d/%m/%y %H:%M:%S.%f')

def get_time_difference(time_begin, time_end):
  return str((time_end - time_begin).total_seconds())

def parse_files(files): 
  for filen in files:
    parse_file(filen)

def parse_file(filen):
  # Get when request was sent from reducer
  begin_re = re.compile("Blocks\sin\smessage\s(.*)\sare\s(.*)")
  send_re = re.compile("Finished\ssending.*[^a]id\s=\s(.*),\ssize")
  recv_re = re.compile("Handling\sbuffer.*[^a]id\s=\s(.*),\ssize")
  ack_re = re.compile("Response\sto.*aid\s=\s(.*),\sid")
  ack_sent_re = re.compile("Finished\ssending.*aid\s=\s(.*),\sid")
  reply_re = re.compile("Handling\sack\smessage.*aid\s=\s(.*),\sid")
  got_block_re = re.compile("Got\sremote\sblock\s(.*)\ssize.*after\s\s(.*)ms")
  for line in filen:
    begin_sr = begin_re.search(line) 
    if begin_sr:
        msg_id = begin_sr.group(1)
        msg_begin_times[msg_id] = get_datetime_from_line(line)
        blocks = begin_sr.group(2).split(',')
        for block in blocks:
            block_message_map[block] = msg_id
            message_block_map[msg_id].append(block) 

    send_sr = send_re.search(line)
    if send_sr:
        msg_id = send_sr.group(1)
        msg_send_times[msg_id] = get_datetime_from_line(line) 

    recv_sr = recv_re.search(line)
    if recv_sr:
        msg_id = recv_sr.group(1)
        msg_recv_times[msg_id] = get_datetime_from_line(line)
 
    ack_sr = ack_re.search(line)
    if ack_sr:
        msg_id = ack_sr.group(1)
        msg_ack_begin_times[msg_id] = get_datetime_from_line(line)

    ack_sent_sr = ack_sent_re.search(line)
    if ack_sent_sr:
        msg_id = ack_sent_sr.group(1)
        msg_ack_end_times[msg_id] = get_datetime_from_line(line)

    reply_sr = reply_re.search(line)
    if reply_sr:
        msg_id = reply_sr.group(1)
        msg_reply_times[msg_id] = get_datetime_from_line(line)
 
    got_block_sr = got_block_re.search(line)
    if got_block_sr:
        block_id = got_block_sr.group(1)
        wall_time = got_block_sr.group(2).strip()
        block_finish_times[block_id] = get_datetime_from_line(line)
        block_wall_times[block_id] = str(float(wall_time)/1000.0)

def print_header():
    print "block_id send_req network build_reply send_reply network parse wall_time" 

def print_shuffle_stats(reducer_id, num_reducers, shuffle_id = 0,
        print_block_stats=False):
    # print_header()
    start = datetime.max
    end = datetime.min
    for i in range(0, num_reducers):
        block = "shuffle_" + str(shuffle_id) + "_" + str(i) + "_" + str(reducer_id) 
        if block in block_message_map: 
                msg_id = block_message_map[block]
                start = min(start, msg_begin_times[msg_id])
                end = max(end, msg_reply_times[msg_id])
                if print_block_stats:
                    print_shuffle_block_stats(block)
    print "shuffle " + str(reducer_id) + " start " + str(start) +\
          " end " + str(end) + " delta " + str((end - start).total_seconds())
        

def print_shuffle_block_stats(shuffle_block):
    # Get msg id first
    msg_id = block_message_map[shuffle_block]
    if msg_id not in msg_recv_times:
        return
    print shuffle_block + "," + str(msg_begin_times[msg_id]) + "," +\
            get_time_difference(msg_begin_times[msg_id], msg_send_times[msg_id]) + "," +\
            get_time_difference(msg_send_times[msg_id], msg_recv_times[msg_id]) + "," +\
            get_time_difference(msg_recv_times[msg_id], msg_ack_begin_times[msg_id]) + "," +\
            get_time_difference(msg_ack_begin_times[msg_id], msg_ack_end_times[msg_id]) + "," +\
            get_time_difference(msg_reply_times[msg_id], block_finish_times[shuffle_block]) + "," +\
            block_wall_times[shuffle_block]

def main():
    parser = argparse.ArgumentParser(description='Parse shuffle logs')
    parser.add_argument('files', metavar='N', type=file, nargs='+',
                        help='log files to parse')
    parser.add_argument('-n', dest='num_reducers', type=int, 
                        required=True, help='total number of reducers')
    parser.add_argument('-r', dest='reducer_id', 
                        default=-1, required=False, type=int,
                        help='restrict shuffle stats to this reducer')
    args = parser.parse_args()
    parse_files(args.files)

    if args.reducer_id != -1:
        print_shuffle_stats(args.reducer_id, args.num_reducers, 0, True)
    else:
        for i in range(0, args.num_reducers): 
            print_shuffle_stats(i, args.num_reducers, 0, False)

if __name__ == "__main__":
    main()
