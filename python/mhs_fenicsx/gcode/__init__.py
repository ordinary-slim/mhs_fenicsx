from __future__ import annotations
import numpy as np
from enum import Enum, IntEnum

class TrackType(Enum):
    PRINTING  = 0
    COOLING   = 1
    DWELLING  = 2
    RECOATING = 3

class Track:
    def __init__(self, p0:np.ndarray, p1:np.ndarray,
                       t0:float,t1:float,
                       track_type:TrackType,
                       speed:np.ndarray,power:float,
                       index=-1,
                       ):
        self.p0 = p0
        self.p1 = p1
        self.t0 = t0
        self.t1 = t1
        self.type = track_type
        self.speed = speed
        self.power = power
        self.index = index
        self.length = np.linalg.norm(p1-p0)

    def get_direction(self):
        step = (self.p1-self.p0)
        return step/np.linalg.norm(step)

    def get_speed(self):
        return self.speed*self.get_direction()

    def get_position(self,time):
        if (time < self.t0 or time > self.t1):
            raise Exception( "Time is out of bounds for this track.")
        return self.p0 + (time-self.t0)/(self.t1-self.t0)*(self.p1-self.p0)

    def __repr__(self):
        return f"Track #{self.index} is a {self.type} track from {self.p0}@t={self.t0} to {self.p1}@t={self.t1}"

class Path:
    def __init__(self,tracks:list[Track]):
        self.tracks = tracks
        self.times = np.empty(len(tracks)+1)
        for idx in range(len(tracks)):
            self.times[idx] = tracks[idx].t0
        self.times[-1] = tracks[-1].t1
        self.current_track = tracks[0]

    def update(self,time):
        self.current_track = self.get_track(time)

    def get_track(self,t:float):
        assert (t >= self.tracks[0].t0 and t <= self.tracks[-1].t1), "Time is out of bounds for this path."
        idx_track = 0
        for idx, track in enumerate(self.tracks):
            if t < track.t1 - 1e-7:
                idx_track = idx
                break
        return self.tracks[idx_track]

    def __repr__(self):
        return str([str(t) for t in self.tracks])

def gcode_to_path(gcodeFile,default_power=100.0):
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
                          previousTime,currentTime,
                          trackType,currentSpeed,currentPower,
                          index=track_counter))
            previousPosition=currentPosition.copy()
            previousTime=currentTime
            track_counter+=1

    gf.close()
    return Path(tracks)

if __name__=="__main__":
    tracks = gcode_to_path("path.gcode")
