import socket
import struct

def parse_netflow_v5_header(data):
    header = struct.unpack('!HHIIIIHH', data[:24])
    return {
        'version': header[0],
        'count': header[1],
        'sys_uptime': header[2],
        'unix_secs': header[3],
        'unix_nsecs': header[4],
        'flow_sequence': header[5],
        'engine_type': header[6] >> 8,
        'engine_id': header[6] & 0xFF,
        'sampling_interval': header[7]
    }

def parse_netflow_v5_record(data):
    records = []
    record_size = 48
    for i in range(0, len(data), record_size):
        if len(data[i:i+record_size]) < record_size:
            continue  # نادیده گرفتن رکورد ناقص

        r = struct.unpack('!IIIHHIIIIHHBBBBHHBBH', data[i:i+record_size])
        record = {
            'src_ip': socket.inet_ntoa(struct.pack('!I', r[0])),
            'dst_ip': socket.inet_ntoa(struct.pack('!I', r[1])),
            'next_hop': socket.inet_ntoa(struct.pack('!I', r[2])),
            'input_if': r[3],
            'output_if': r[4],
            'packets': r[5],
            'bytes': r[6],
            'start_time': r[7],
            'end_time': r[8],
            'src_port': r[9],
            'dst_port': r[10],
            'tcp_flags': r[12],
            'protocol': r[13],
            'tos': r[14]
        }
        records.append(record)
    return records


def start_netflow_server(ip="0.0.0.0", port=2055):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    print(f"NetFlow v5 Collector listening on {ip}:{port}...")

    while True:
        data, addr = sock.recvfrom(65535)
        if len(data) < 24:
            continue

        header = parse_netflow_v5_header(data[:24])
        print(f"\n--- Received NetFlow v5 packet from {addr[0]} ---")
        print(f"Flow Count: {header['count']} | Sequence: {header['flow_sequence']}")

        records = parse_netflow_v5_record(data[24:])
        for i, rec in enumerate(records):
            print(f"Flow {i+1}: {rec['src_ip']}:{rec['src_port']} → {rec['dst_ip']}:{rec['dst_port']} "
                  f"Protocol={rec['protocol']} Bytes={rec['bytes']} Packets={rec['packets']}")

start_netflow_server()
