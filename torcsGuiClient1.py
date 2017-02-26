#!/usr/bin/env python
'''
Created on Apr 4, 2012

@author: lanquarden
'''

# @author: Ahmed Nassar
# This code builds on "lanquarden" code from:
# https://github.com/lanquarden/pyScrcClient
# by adding mapping capabilities based on
# guidelines and code snippets from this presentation:
# "Virtual robotic car racing with Python and TORCS"
# https://www.youtube.com/watch?v=BGOtqXA_y1E

import sys
import argparse
import socket
import driver

import time 
import math
import threading
from threading import Thread
from PyQt4 import QtGui, QtCore
from collections import deque




def main():
    arguments = parseArguments()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error, msg:
        print '*** Could not make a socket.'
        sys.exit(-1)
    # one second timeout
    sock.settimeout(1.0)

    d = driver.Driver(arguments.stage)

    app = QtGui.QApplication(sys.argv)
    view = TorcsTrackView()
    thr1 = Thread(target = repaintTrack, args = (view, ))
    thr2 = Thread(target = launchClient, args = (view, arguments, d, sock, ))

    try:
        thr1.start()
        thr2.start()
    except:
        print "Error: unable to start thread"
    sys.exit(app.exec_())
    



###########################################################
class RangeFinderRay:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


###########################################################
class TorcsTrackView(QtGui.QWidget):
    
    def __init__(self):
        super(TorcsTrackView, self).__init__()
        self.offsetX     = 700;
        self.offsetY     = 500;
        self.trackLength = 1908.32;
        self.standardTrackLength = 1908.32; # Length of oval A-Speedway track
        self.mult       = 0.5 * (self.standardTrackLength/self.trackLength);
        self.curX       = 0;
        self.curY       = 0;
        self.curTheta   = 0;
        self.curLapTime = 0;
        self.CAR_LENGTH = 0.135;
        self.count      = 0;
        self.rays       = deque([], 10000);
        self.ranges     = deque([], 20);
        self.cx     = 200;
        self.cy     = 400;
        self.mDash  = 0.75;
        self.rDash  = 150;
        self.lock = threading.Lock();
        self.initUI();
        
    def initUI(self):
        self.setGeometry(50, 50, 800, 400)
        self.setWindowTitle('Torcs Track View')
        self.show()

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.blue)
        self.setPalette(p)
        self.drawTrack(qp)
        self.drawDashBoard(qp);
        qp.end()
        
    def drawTrack(self, qp):
        pen = QtGui.QPen(QtCore.Qt.white, 3, QtCore.Qt.SolidLine)
        qp.setBrush(QtGui.QColor(0, 200, 0))
        qp.setPen(pen)
        qp.setOpacity(0.9)
        qp.setRenderHint( QtGui.QPainter.Antialiasing )

        # Draw rays
        dx = self.offsetX;
        dy = self.offsetY;
        m  = self.mult;
        with self.lock:
            #print len(self.rays);
            for r in self.rays:
                #line = QtCore.QLineF(dx + m*r.x1, dy + m*r.y1, dx + m*r.x2, dy + m*r.y2)
                #qp.drawLine(line);
                qp.drawPoint(dx + m*r.x2, dy + m*r.y2);

    def drawDashBoard(self, qp):
        pen = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.SolidLine)
        qp.setBrush(QtGui.QColor(0, 200, 0))
        qp.setPen(pen)
        qp.setOpacity(0.5)
        qp.setRenderHint( QtGui.QPainter.Antialiasing )

        center = QtCore.QPoint(self.cx, self.cy);
        qp.setBrush(QtCore.Qt.yellow);
        qp.drawEllipse(center, self.rDash, self.rDash);
        qp.setOpacity(0.95)

        # Draw ranges
        with self.lock:
            i = 0;
            for r in self.ranges:
                line = QtCore.QLineF(self.cx, self.cy, 0, 0);
                line.setAngle(10*i - 90);
                line.setLength( self.mDash * r );
                qp.drawLine(line);
                i += 1;


    ##############################################
    # Bicycle Model
    ##############################################
    # Define the path as a function of its length s.
    # Theta is the angle between the path tangent at s and the global x axis.
    def updateLocation( self, speed, dt, steeringAngle ):
        ds = speed * dt; # Distance traveled in one time step dt.
        #turnAngle = ds * math.tan( steeringAngle );
        #if ( abs( turnAngle ) < 0.00000001 ):
        steeringAngle = steeringAngle * math.pi / 180.0;
        if ( abs( steeringAngle ) < 0.000001 ):
            nextX = self.curX + ds * math.cos( self.curTheta );
            nextY = self.curY + ds * math.sin( self.curTheta );
            self.curX     = nextX;
            self.curY     = nextY;
            self.curTheta = self.curTheta;
            #print 'curX={0:.3f}, curY={1:.3f}, curTheta={2:.3f}'.format( self.curX, self.curY, self.curTheta );
        else:
            radius = self.CAR_LENGTH / math.tan( steeringAngle );
            beta = ds / radius;
            #print 'radius={0:.3f}, speed={1:.3f}, dt={2:.3f}, ds={3:.3f}, beta={4:.6f}'.format(
            #    radius, speed, dt, ds, beta );
            cx = self.curX - radius * math.sin( self.curTheta );
            cy = self.curY + radius * math.cos( self.curTheta );
            nextTheta = self.curTheta + beta;
            nextX = cx + radius * math.sin( nextTheta );
            nextY = cy - radius * math.cos( nextTheta );
            self.curX     = nextX;
            self.curY     = nextY;
            self.curTheta = nextTheta;


    def updateTrack(self, driver):
        with self.lock:
            sx = driver.state.speedX;
            sy = driver.state.speedY;
            trackSpeed = sx * math.cos( driver.state.angle ) + sy * math.sin( driver.state.angle );
            trackSpeed = (1.0/3.6) * trackSpeed; # km/h to m/s
            dt = (driver.state.curLapTime - self.curLapTime);
            ds = trackSpeed * dt;

            steeringAngle = - driver.steeringAngle;
            self.updateLocation( trackSpeed, dt, steeringAngle );
            #print self.curTheta;

            self.curLapTime = driver.state.curLapTime;

            N = len(driver.state.track);
            for i in range( 0, N ):
                self.ranges.append( driver.state.track[i] );

            self.count += 1;
            if ( self.count < 10 ):
                return;
            self.count = 0;
            for i in range( 0, N ):
                x2 = self.curX + driver.state.track[i] * math.cos( self.curTheta + i*math.pi/N - math.pi/2 );
                y2 = self.curY + driver.state.track[i] * math.sin( self.curTheta + i*math.pi/N - math.pi/2 );
                self.rays.append( RangeFinderRay( self.curX, self.curY, x2, y2 ) );


def repaintTrack( view ):
    while(True):
        time.sleep(0.05)
        view.update()


###########################################################
def parseArguments():
	# Configure the argument parser
	parser = argparse.ArgumentParser(description = 'Python client to connect to the TORCS SCRC server.')

	parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
						help='Host IP address (default: localhost)')
	parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
						help='Host port number (default: 3001)')
	parser.add_argument('--id', action='store', dest='id', default='SCR',
						help='Bot ID (default: SCR)')
	parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
						help='Maximum number of learning episodes (default: 1)')
	parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
						help='Maximum number of steps (default: 0)')
	parser.add_argument('--track', action='store', dest='track', default=None,
						help='Name of the track')
	parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
						help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
	arguments = parser.parse_args()

	# Print summary
	print 'Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port
	print 'Bot ID:', arguments.id
	print 'Maximum episodes:', arguments.max_episodes
	print 'Maximum steps:', arguments.max_steps
	print 'Track:', arguments.track
	print 'Stage:', arguments.stage
	print '*********************************************'
	return arguments;
###########################################################

def launchClient(view, arguments, d, sock):
	shutdownClient = False
	curEpisode     = 0
	verbose        = False

	while not shutdownClient:
		while True:
			print 'Sending id to server: ', arguments.id
			buf = arguments.id + d.init()
			print 'Sending init string to server:', buf
			
			try:
				sock.sendto(buf, (arguments.host_ip, arguments.host_port))
			except socket.error, msg:
				print "Failed to send data...Exiting..."
				sys.exit(-1)
				
			try:
				buf, addr = sock.recvfrom(1000)
			except socket.error, msg:
				print "didn't get response from server..."
		
			if buf.find('***identified***') >= 0:
				print 'Received: ', buf
				break

		currentStep = 0
		
		while True:
			# wait for an answer from server
			buf = None
			try:
				buf, addr = sock.recvfrom(1000)
			except socket.error, msg:
				print "didn't get response from server..."
			
			if verbose:
				print 'Received: ', buf
			
			if buf != None and buf.find('***shutdown***') >= 0:
				d.onShutDown()
				shutdownClient = True
				print 'Client Shutdown'
				break
			
			if buf != None and buf.find('***restart***') >= 0:
				d.onRestart()
				print 'Client Restart'
				break
			
			currentStep += 1
			if currentStep != arguments.max_steps:
				if buf != None:
					buf = d.drive(buf)
					view.updateTrack( d );
			else:
				buf = '(meta 1)'
			
			if verbose:
				print 'Sending: ', buf
			
			if buf != None:
				try:
					sock.sendto(buf, (arguments.host_ip, arguments.host_port))
				except socket.error, msg:
					print "Failed to send data...Exiting..."
					sys.exit(-1)
		
		curEpisode += 1
		
		if curEpisode == arguments.max_episodes:
			shutdownClient = True

	sock.close()



if __name__ == '__main__':
    main()


