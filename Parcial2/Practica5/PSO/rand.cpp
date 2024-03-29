/* Rutina para generacion de numeros aleatorios gaussianos con
   media cero y desviacion estandar sigma.

   Codigo original de Joachim Sprave, University of Dortmund, 
   Alemania, Dept. of Computer Science and Systems Analysis

*/


#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include "rand.h"

#define NRAND_SAMPLES 5
#define Uniform randreal

using namespace std;


double Gauss(double sigma)
{
	/* System generated locals */
	double ret_val;
	
	/* Local variables */
	static double d, u, x, y, u0, u1, u2;
	
	
	/* 	SIGMA	--> standard deviation */
	
	/* L1: */
	u = Uniform();
	u0 = Uniform();
    
	if (u >= .919544406) 
	{
		goto L2;
	}
	x = (u0 + u * .825339283) * 2.40375766 - 2.11402808;
	goto L10;
	L2:
	if (u < .965487131) {
		goto L4;
	}
	L3:
	u1 = Uniform();
	y = sqrt(4.46911474 - log(u1) * 2.);
	u2 = Uniform();
	if (y * u2 > 2.11402808) {
		goto L3;
	}
	goto L9;
	L4:
	if (u < .949990709) {
		goto L6;
	}
	L5:
	u1 = Uniform();
	y = u1 * .273629336 + 1.84039875;
	u2 = Uniform();
	if (exp(y * -.5 * y) * .39894228 - .443299126 + y * .209694057 < u2 *
		.0427025816) {
		goto L5;
	}
	goto L9;
	L6:
	if (u < .925852334) {
		goto L8;
	}
	L7:
	u1 = Uniform();
	y = u1 * 1.55066917 + .289729574;
	u2 = Uniform();
	if (exp(y * -.5 * y) * .39894228 - .443299126 + y * .209694057 < u2 *
		.0159745227) {
		goto L7;
	}
	goto L9;
	L8:
	u1 = Uniform();
	y = u1 * .289729574;
	u2 = Uniform();
	if (exp(y * -.5 * y) * .39894228 - .382544556 < u2 * .0163977244) {
		goto L8;
	}
	L9:
	x = y;
	if (u0 >= .5) {
		x = -y;
	}
	L10:
	ret_val = sigma * x;
	return ret_val;
}

double NI(double m, double sigma)
{
    return m + Gauss(sigma);
}

void initrandom(int seed)
{
    Seed = seed;
}

double randreal(void)
{
    double result;

    if (!isinit) {
	srand(Seed);
	isinit = 1;
    }
    result = ((double) rand());
    result /= RAND_MAX;

    return (result);
}

double rndreal(double lo,double hi)
{
  double ret;
  ret=(lo + randreal() * (double)(hi - lo ));
   return ret;
}
int 
flip(double rate){
	if(randreal()<rate) return 1;
	return 0;
}

int 
rndint(int lo, int hi)
{
    return (int)(lo + randreal() * (hi - lo + 1));
}


/*
int main()
{

  initrandom(time(0));

  cout<<"distribución uniforme usando rndreal"<<endl;
  for(int i=0;i<10;i++)
    cout<<rndint(-20,20)<<endl;
  
  cout<<"Real"<<endl;
  for(int i=0;i<10;i++)
    cout<<rndreal(-20,20)<<endl;


  cout<<"distribución gausiana, con media 0 y desviación estándard 1"<<endl;
  for(int i=0;i<10;i++)
    cout<<NI(0,1)<<endl;
}

*/
