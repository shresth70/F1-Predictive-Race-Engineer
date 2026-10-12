import socket, struct, threading, time
from collections import deque

# F1 24 UDP packet header (2024): packet format/version fields, session UID,
# session time, frame identifier, player/car indices. Packet ID 6 = CarTelemetry.
# The 2024 CarTelemetryData record is 60 bytes; this receiver intentionally reads
# only stable leading fields after the 29-byte header and marks unsupported packets.
class TelemetryReceiver:
    def __init__(self):
        self.sock=None; self.thread=None; self.running=False; self.port=20777
        self.samples=deque(maxlen=300); self.latest=None; self.message="Listener stopped"; self.lock=threading.Lock()
    def start(self, port=20777):
        if self.running: return self.status()
        self.port=port
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(("0.0.0.0",port)); sock.settimeout(1.0)
        self.sock=sock; self.running=True; self.message=f"Listening on UDP :{port}; waiting for F1 24 packets"
        self.thread=threading.Thread(target=self._loop,daemon=True); self.thread.start()
        return self.status()
    def stop(self):
        self.running=False
        if self.sock:
            try: self.sock.close()
            except OSError: pass
        self.sock=None; self.message="Listener stopped"
        return self.status()
    def _loop(self):
        while self.running and self.sock:
            try: data, addr=self.sock.recvfrom(4096)
            except socket.timeout: continue
            except OSError: break
            try: sample=self._parse(data,addr)
            except (ValueError, struct.error, IndexError): continue
            if sample:
                with self.lock:
                    self.latest=sample; self.samples.append(sample)
                    self.message=f"Receiving F1 24 telemetry from {addr[0]}"
        self.running=False
    def _parse(self, data, addr):
        if len(data)<29: return None
        # Packet header offsets in F1 24: packetFormat (u16), gameYear, gameMajor,
        # gameMinor, packetVersion, packetId, sessionUID (u64), sessionTime (f32),
        # frameIdentifier (u32), overallFrameIdentifier (u32), playerCarIndex, secondaryIndex.
        fmt,year,major,minor,version,pid=struct.unpack_from("<HBBBBB",data,0)
        if fmt != 2024 or pid != 6 or len(data)<29+60*22+2: return None
        session_time=struct.unpack_from("<f",data, session_offset:=session_time_offset())[0]
        player=data[27]
        record_offset=29+player*60
        speed, throttle, steer, brake, clutch, gear, rpm = struct.unpack_from("<HfffB b H", data, record_offset)
        return {"timestamp":round(time.time(),3),"session_time":round(session_time,3),"packet_format":fmt,
                "player_car_index":player,"speed_kph":int(speed),"throttle":round(float(throttle),3),
                "steering":round(float(steer),3),"brake":round(float(brake),3),"gear":int(gear),
                "rpm":int(rpm),"source_ip":addr[0]}
    def snapshot(self):
        with self.lock:
            return {"connected":bool(self.running and self.latest),"listening":self.running,
                    "message":self.message,"latest":self.latest,"samples":list(self.samples)}
    def status(self):
        return {"listening":self.running,"port":self.port,"message":self.message,
                "sample_count":len(self.samples),"connected":bool(self.running and self.latest)}

def session_time_offset():
    return session_time_offset_value()
def session_time_offset_value():
    # Header layout: 2 + 5 bytes + 8-byte UID = 15; session time at byte 15.
    return 15
