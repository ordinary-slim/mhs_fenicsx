from __future__ import annotations
import numpy as np
import numpy.typing as npt
from enum import Enum, IntEnum

tol = 1e-9

class TrackType(Enum):
    PRINTING  = 0
    COOLING   = 1
    DWELLING  = 2
    RECOATING = 3

class Track:
    def __init__(self, p0:npt.NDArray[np.float64], p1:npt.NDArray[np.float64],
                       t0:np.float64, t1:np.float64,
                       track_type:TrackType,
                       speed:np.float64, power:np.float64,
                       ):
        self.p0 = p0
        self.p1 = p1
        self.t0 = t0
        self.t1 = t1
        self.type = track_type
        self.speed = speed
        self.power = power
        self.length = np.linalg.norm(p1-p0)

    def get_direction(self):
        step = (self.p1-self.p0)
        stepsize = np.linalg.norm(step)
        if stepsize >1e-9:
            return step/stepsize
        else:
            return np.zeros(3, dtype=np.float64)

    def get_speed(self):
        return self.speed*self.get_direction()

    def get_position(self,time, bound=False):
        if ((time < self.t0 or time > self.t1 + tol) and not(bound)):
            raise Exception( "Time is out of bounds for this track.")
        if time < self.t0:
            return self.p0.copy()
        elif time > self.t1:
            return self.p1.copy()
        else:
            return self.p0 + (time-self.t0)/(self.t1-self.t0)*(self.p1-self.p0)

    def __repr__(self):
        return f"{self.type} track from {self.p0}@t={self.t0} to {self.p1}@t={self.t1}"

def get_infinite_track(p0:npt.NDArray[np.float64],
                       t0:np.float64,
                       speed:npt.NDArray[np.float64],
                       power:np.float64):
    dt = 1e9
    t1 = t0 + dt
    p1 = p0 + dt * speed
    return Track(p0, p1, t0, t1, TrackType.PRINTING, np.linalg.norm(speed), power)

class Path:
    def __init__(self,tracks:list[Track]):
        self.tracks = tracks
        self.times = np.empty(len(tracks)+1, dtype=np.float64)
        for idx in range(len(tracks)):
            self.times[idx] = tracks[idx].t0
        self.times[-1] = tracks[-1].t1
        self.current_track = tracks[0]

    def final_time(self) -> float:
        return self.times[-1]

    def update(self,time):
        self.current_track = self.get_track(time)

    def get_track_idx(self, t:float, pad=-1e-9):
        assert (t >= self.tracks[0].t0 and t <= self.tracks[-1].t1 + tol), \
                "Time is out of bounds for this path."
        idx_track = 0
        for track in self.tracks:
            if t < track.t1 + pad:
                break
            idx_track += 1
        idx_track = min(idx_track,len(self.tracks)-1)
        return idx_track

    def get_track(self,t:float, pad=-1e-9):
        return self.tracks[self.get_track_idx(t, pad=pad)]

    def get_track_interval(self, t0:np.float64, t1:np.float64):
        assert(t0 <= t1)
        return self.tracks[self.get_track_idx(t0): self.get_track_idx(t1, pad=+1e-9)+1]

    def __repr__(self):
        return str([str(t) for t in self.tracks])

def gcode_to_path(gcodeFile,default_power=100.0) -> Path:
    class Index(IntEnum):
        X = 0
        Y = 1
        Z = 2

    gf = open( gcodeFile, 'r' )

    tracks = []
    track_counter = 0
    firstPositionRead = False
    previousPosition = np.zeros(3)
    previousTime = 0.0
    previousNonZeroSpeed = 10

    for rawline in gf:
        timePassed = False
        trackType = TrackType.COOLING
        currentPosition = previousPosition.copy()
        currentTime = previousTime
        currentPower = 0.0
        currentSpeed = previousNonZeroSpeed
        hasMotion=False
        line = rawline.rstrip("\n")
        line = line.split(";", 1)[0]# remove comments

        instructions = line.split()
        for instruction in instructions:
            instructionType = instruction[0]
            instructionValue = float(instruction.lstrip(instructionType))
            if instructionType == "G":
                if instructionValue == 4:
                    trackType = TrackType.DWELLING
            elif instructionType == "F":
                currentSpeed = instructionValue
                previousNonZeroSpeed = currentSpeed
            elif instructionType in ["X", "Y", "Z"]:
                currentPosition[ int( Index[instructionType] ) ] = instructionValue
                hasMotion=True
            elif instructionType == "E":
                if instructionValue > 0.0:
                    currentPower = default_power
                    trackType = TrackType.PRINTING
            elif instructionType == "P":
                if trackType in [TrackType.DWELLING, TrackType.RECOATING]:
                    currentTime += instructionValue
                    timePassed=True
            elif instructionType == "R":
                if (trackType == TrackType.DWELLING) and (instructionValue > 0):
                    trackType = TrackType.RECOATING
        if trackType in [TrackType.DWELLING or TrackType.RECOATING]:
            currentSpeed = 0.0

        if (hasMotion):
            if (firstPositionRead):
                currentTime = previousTime + np.linalg.norm(currentPosition - previousPosition) / currentSpeed
                timePassed = True
            else:
                firstPositionRead = True
                previousPosition = currentPosition.copy()

        if timePassed:
            tracks.append(Track(previousPosition,currentPosition,
                          previousTime, currentTime,
                          trackType,currentSpeed,currentPower))
            previousPosition=currentPosition.copy()
            previousTime=currentTime
            track_counter+=1

    gf.close()
    return Path(tracks)

if __name__=="__main__":
    tracks = gcode_to_path("path.gcode")
