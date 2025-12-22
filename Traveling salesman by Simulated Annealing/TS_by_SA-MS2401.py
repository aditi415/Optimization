 from scipy import *
 from numpy import *
 from pylab import *
 from geopy.geocoders import Nominatim
 import random

def Distance(P1, P2):
  	if(P1 == P2):
		 return 0.0
 	d = sqrt( (P1[0] - P2[0])**2 + (P1[1] - P2[1])**2 ) 
return d


 def TotalDistance(P, seq):
 	dist=0.0
 	N = len(seq)
 	for i in range(N-1):
 		dist += Distance(P[seq[i]], P[seq[i+1]])
 		dist += Distance(P[seq[N-1]], P[seq[0]])
 	return dist
 

def readCities(PNames):
 	P=[]

 	geolocator = Nominatim(user_agent="MyApp")
 	j = 0
 	with open("./india_cities.txt") as file:
 
	for line in file:
 	city = line.rstrip('\n')
 	if(city == ""):
	 	break
 	theLocation = city + ", India"
 	pt = geolocator.geocode(theLocation, timeout=10000)
 	y = round(pt.latitude, 2)
 	x = round(pt.longitude, 2)
 	print ("City[%2d] = %s (%5.2f, %5.2f)" %(j, city, x, y))
 	P.insert(j, [x, y])
 	PNames.insert(j, city)
 	j += 1



def Plot(seq, P, dist, PNames):
 
 Pt = [P[seq[i]] for i in range(len(seq))]
 Pt += [P[seq[0]]]
 Pt = array(Pt)
 	title('Total distance = ' + str(dist))
 plot(Pt[:, 0], Pt[:, 1], '-o')
 for i in range(len(P)):
 	annotate(PNames[i], (P[i][0], P[i][1]))
 show()
 

def swap(P, seq, dist, N1, N2, temp, nCity):
	 N1L = N1 - 1
 	if(N1L < 0):
		 N1L += nCity
	 N1R = N1 + 1
	 if(N1R >= nCity):
		 N1R = 0
	 N2L = N2 - 1
	 if(N2L < 0):
 		N2L += nCity
 	N2R = N2 + 1
	 if(N2R >= nCity):
		 N2R = 0
 

I1 = seq[N1]
 I2 = seq[N2]
 I1L = seq[N1L]
 I1R = seq[N1R]
 I2L = seq[N2L]
 I2R = seq[N2R]


 delta = 0.0
 delta += Distance(P[I1L], P[I2])
 delta += Distance(P[I1], P[I2R])
 delta -= Distance(P[I1L], P[I1])
 delta -= Distance(P[I2], P[I2R])


 if(N1 != N2L and N1R != N2 and N1R != N2L and N2 != N1L):
 	delta += Distance(P[I2], P[I1R])
 	delta += Distance(P[I2L], P[I1])
 	delta -= Distance(P[I1], P[I1R])
 	delta -= Distance(P[I2L], P[I2])

 prob = 1.0
 if(delta > 0.0):
 	prob = exp(-delta/temp)
 
rndm = random.random()
 
 
if(rndm < prob):
 	dist += delta
 
	seq[N1] = I2
 	seq[N2] = I1
 
	dif = abs(dist - TotalDistance(P, seq))
 	if(dif*dist > 0.01):
 		dist = TotalDistance(P, seq)
		 print("%s\n" %("in SWAP -->"))
		 print( "N1=%3d N2=%3d N1L=%3d N1R=%3d N2L=%3d N2R=%3d \n" % (N1,
 N2, N1L, N1R, N2L, N2R) )
		 print( "I1=%3d I2=%3d I1L=%3d I1R=%3d I2L=%3d I2R=%3d \n" % (I1,
 I2, I1L, I1R, I2L, I2R) )
 		print( "T= %f D= %f delta= %f p= %f rn= %f\n" % (temp, dist,delta, prob, rndm) )
 		print(seq)
 		print("%s\n" % ("") )

		 input("Press Enter to continue...")
 
		return dist, True
 else:
 		return dist, False
 


def reverse(P, seq, dist, N1, N2, temp, nCity):
 	N1L = N1 - 1
 	if(N1L < 0):
 		N1L += nCity
 
	N2R = N2 + 1
	if(N2R >= nCity):
 		N2R = 0
 
	delta = 0.0
	if( (N1 != N2R) and (N2 != N1L) ):
 		delta =   Distance(P[seq[N1L]], P[seq[N2]]) \
                + Distance(P[seq[N1]], P[seq[N2R]]) \
                - Distance(P[seq[N1L]], P[seq[N1]]) \
                - Distance(P[seq[N2]], P[seq[N2R]])
 	else:
 		return dist, False
 
	prob = 1.0
 	if(delta > 0.0):
 		prob = exp(-delta/temp)
 
	rndm = random.random()
 
 
	 if(rndm < prob):
 		dist += delta
 		i = N1
 		j = N2
 		while(i < j):
 			u = seq[i]
 			seq[i] = seq[j]
 			seq[j] = u
 			i += 1
 			j -= 1
 
		dif = abs(dist - TotalDistance(P, seq))
 		if(dif*dist > 0.01):
 			print("in REVERSE N1L=%3d  N2R=%3d \n" % (N1L, N2R) )
 			print( "N1=%3d N2=%3d  T= %f D= %f delta= %f p= %f rn= %f\n" %
 (N1, N2, temp, dist, delta, prob, rndm) )
 			print(seq)
 			print()
 
 		return dist, True
 else:
 		return dist, False


if __name__=='__main__':
 	 PNames=[] 

	 P = readCities(PNames)
 	 nCity = len(P)
 
	 maxTsteps = 250
 	 fCool = 0.9
 	 maxSwaps = 2000
 
 	 maxAccepted = 10*nCity 
	 seq= arange(0, nCity, 1)
 	dist = TotalDistance(P, seq)
 	temp = 10.0 * dist
 	print("\n\n")
 	print(seq)
	 print("\n nCity= %3d  dist= %f  temp= %f \n" % (nCity, dist, temp) )
 	Plot(seq, P, dist, PNames)


 oldDist = 0.0
 convergenceCount = 0

 for t in range(1, maxTsteps+1):
 	if(temp < 1.0e-6):
 		break
 	accepted = 0
 	iteration = 0
 
	 while((iteration <= maxSwaps) and (accepted < maxAccepted) ):
 	 while(iteration <= maxSwaps):
 
	N1 = -1
 	while(N1 < 0 or N1 >= nCity):
 		N1 = ((int) (random.random() * 1000.0)) % nCity
 	
	N2 = -1
 	while(N2 < 0 or N2 >= nCity or N2 == N1):
 		N2 = ((int) (random.random() * 1000.0)) % nCity
 
	if(N2 < N1):
 		N1= N1 + N2
 		N2 = N1 - N2
 		N1 = N1 - N2
 	print(iteration, "N  ", N1, N2)
 
	chk = random.uniform(0, 1)
 	if( (chk < 0.5) and (N1+1 != N2) and (N1 != ((N2+1) % nCity)) ):
 		dist, flag = swap(P, seq, dist, N1, N2, temp, nCity)
 else :
 		dist, flag = reverse(P, seq, dist, N1, N2, temp, nCity)
 
	if(flag):
 		accepted += 1
 		iteration += 1


 print("Iteration: %d  temp=%f  dist=%f" %(t, temp, dist))
 print("seq = ")
 set_printoptions(precision=3)
 print(seq)
 print("%c%c" % ('\n', '\n'))
 
if(abs(dist - oldDist) < 1.0e-4):
 	convergenceCount += 1
 else:
 	convergenceCount = 0
 if(convergenceCount >= 4):
 	break
 if((t%25) == 0): 
	Plot(seq, P, dist, PNames)
 
	temp *= fCool
 	oldDist = dist
 

Plot(seq, P, dist, PNames)
