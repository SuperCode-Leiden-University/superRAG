#include <iostream>	
#include <cmath>	
#include <fstream>

#include "ode.h"

using namespace std;

void dYdt (double, double *, double *);
void acc_func(double *, double *, double *);
void solution(double , double *, double *);


void energy_coord(double , double , double , double , double &, double &, double &, double &, double &);

void NormAng(double &);

#define SESSION 4
// 0 = general study
// 1 = small angles
// 2 = convergence_angle
// 3 = flip and chaos
// 4 = chaos-bis (fluctuation size)

static double g_L1=1., g_m1=1., g_L2=1., g_m2=1.; // length and mass of the pendulum

// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------

int main (){
  double g=9.81, L1=g_L1, m1=g_m1, L2=g_L2, m2=g_m2;
  
  int k=0, N=1000, n_var=2, neq=2*n_var; // deq=2=degree of differential equations, var=varing variables, assuming deq is the same for each variable
  double Ysol[neq], Y[neq], X[n_var], V[n_var]; // X = variabiables, V = dX/dt
  
  double dt=0.001, t, E0, E, E_sol; // N=number of steps to be done, E=energy
  double x1, x2, y1, y2, v1, v2; // position in space (cartesian coord) and angular speed times length of the rod
  double x1_sol, x2_sol, y1_sol, y2_sol, v1_sol, v2_sol;
  
  
  
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  #if SESSION == 0 // ------------------------- GENERAL STUDY: -------------------------
  // -----------------------------------------------------------------------------------
  
  double Y0[neq]={M_PI*0.25, M_PI*0.25, 0., 0.}; // Y0={theta1, theta2, v_theta1, v_theta2} -> initial values (border condition), R=dY/dt
  // NB: thetas are calculated from the vertical, if theta1=theta2=2*pi/4, then the centre and the masses form a line parallel to the ground at time t=0
  
  N=50*N;
  
  ofstream fdata;         // declare Output stream class to operate on files
  fdata.open("dati.dat"); // open output file
  
  cout<<"\t ##### SESSION 0 - general study #####\n"<<endl;
  
  // RK methods ---------------------
  for (int j=1; j<=2; j++){
    for (int i=0; i<neq; i++) Y[i]=Y0[i]; // Y must contein initial values when you start integrating
    t=0; // starting time    
    for (int i=0; i<N; i++){
      
      if (j==1) {
        RK4Step (t, Y, dYdt, dt, neq);
      } else {
        RK2Step (t, Y, dYdt, dt, neq);
      }
      t+=dt;
      
      for (int l=0; l<n_var; l++) NormAng(Y[l]); // this seems useless, but if I plot the graph I want thetas to stay in [ -pi, +pi ]
      
      energy_coord(Y[0], Y[1], Y[2], Y[3], x1, x2, y1, y2, E);
      if (i==0) E0=E;
      
      // t, theta, v_theta, error, x=sin(theta), y=cos(theta)
      fdata<< t <<" "<< Y[0] <<" "<< Y[1] <<" "<< Y[2] <<" "<< Y[3] <<" "<< E-E0 <<" "
        << x1 <<" "<< y1 <<" "<< x2 <<" "<< y2 << endl;
      
    }
    cout<<"RK"<<j*2<<" method completed"<<endl;
    fdata<<"\n"<<endl;
  }
  
  
  
  // Verlet methods ---------------------
  for (int j=1; j<=2; j++){
    for (int i=0; i<n_var; i++) {
      X[i]=Y0[i]; // X,V must contein initial values when you start integrating
      V[i]=Y0[i+n_var];
    }    
    t=0; // starting time
    
    for(int i=0; i<N; i++){ 
      
      if (j==1) {
        p_Verlet (X, V, acc_func, dt, n_var);
      } else {
        v_Verlet (X, V, acc_func, dt, n_var);
      } 
      t+=dt;    
      
      for (int l=0; l<n_var; l++) NormAng(X[l]); // this seems useless, but if I plot the graph I want thetas to stay in [ -pi, +pi ]
      
      energy_coord(X[0], X[1], V[0], V[1], x1, x2, y1, y2, E);
      if (i==0) E0=E;
      
      // t, theta, v_theta, error, x=sin(theta), y=cos(theta)
      fdata<< t <<" "<< X[0] <<" "<< X[1] <<" "<< V[0] <<" "<< V[1] <<" "<< E-E0 <<" "
        << x1 <<" "<< y1 <<" "<< x2 <<" "<< y2 << endl;
        
    }
    if (j==1) {
      cout<<"p_Verlet method completed"<<endl;
    } else {
      cout<<"v_Verlet method completed"<<endl;
    }
    fdata<<"\n"<<endl;
  }
  
  fdata.close();
  
  
  
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  #elif SESSION == 1 // ------------------------- SMALL ANGLES: ------------------------
  // -----------------------------------------------------------------------------------
  
  double Y0[neq]={0., 0.001, 0., 0.};
  
  m2=m1; // solution with both equal to simplify
  L2=L1;
  
  N=3*N;
  
  ofstream fdata;         // declare Output stream class to operate on files
  fdata.open("dati_small.dat"); // open output file
  
  
  cout<<"\t ##### SESSION 1 - small angles #####\n"<<endl;
  
  // RK methods ---------------------
  for (int j=1; j<=2; j++){
    
    for (int i=0; i<neq; i++) Y[i]=Y0[i]; // Y must contein initial values when you start integrating
    t=0; // starting time    
    for (int i=0; i<N; i++){
      
      if (j==1) {
        RK4Step (t, Y, dYdt, dt, neq);
      } else {
        RK2Step (t, Y, dYdt, dt, neq);
      }
      t+=dt;
      solution(t, Y0, Ysol);
      
      for (int l=0; l<n_var; l++) NormAng(Y[l]); // this seems useless, but if I plot the graph I want thetas to stay in [ -pi, +pi ]
      
      energy_coord(Y[0], Y[1], Y[2], Y[3], x1, x2, y1, y2, E);
      if (i==0) E0=E;
      energy_coord(Ysol[0], Ysol[1], Ysol[2], Ysol[3], x1_sol, x2_sol, y1_sol, y2_sol, E_sol);
      
      fdata<< t <<" "<< dt <<" "
        << E-E_sol <<" "<< Y[0]-Ysol[0] <<" "<< Y[1]-Ysol[1] <<" "<< Y[2]-Ysol[2] <<" "<< Y[3]-Ysol[3] << " "
        << E <<" "<< E_sol <<" "<< Y[0] <<" "<< Ysol[0] <<" "<< Y[1] <<" "<< Ysol[1] <<" "<< Y[2] <<" "<< Ysol[2] <<" "<< Y[3] <<" "<< Ysol[3] << endl;
        // the last line is useful to check the sovrapposion in a qualitative/intuitive way
    }
    
    fdata<<"\n"<<endl;
  
  }
  
  
  
  // Verlet methods ---------------------
  for (int j=1; j<=3; j++){
    
    for (int i=0; i<n_var; i++) {
      X[i]=Y0[i]; // X,V must contein initial values when you start integrating
      V[i]=Y0[i+n_var];
    }    
    t=0; // starting time
    
    for(int i=0; i<N; i++){
      
      if (j==1) {
        p_Verlet (X, V, acc_func, dt, n_var);
      } else {
        v_Verlet (X, V, acc_func, dt, n_var);
      }
      t+=dt;
      solution(t, Y0, Ysol);
     
      for (int l=0; l<n_var; l++) NormAng(X[l]); // this seems useless, but if I plot the graph I want thetas to stay in [ -pi, +pi ]
      
      energy_coord(X[0], X[1], V[0], V[1], x1, x2, y1, y2, E);
      if (i==0) E0=E;
      energy_coord(Ysol[0], Ysol[1], Ysol[2], Ysol[3], x1_sol, x2_sol, y1_sol, y2_sol, E_sol);
      
      fdata<< t <<" "<< dt <<" "
        << E-E_sol <<" "<< X[0]-Ysol[0] <<" "<< X[1]-Ysol[1] <<" "<< V[0]-Ysol[2] <<" "<< V[1]-Ysol[3] << " "
        << E <<" "<< E_sol <<" "<< X[0] <<" "<< Ysol[0] <<" "<< X[1] <<" "<< Ysol[1] <<" "<< V[0] <<" "<< Ysol[2] <<" "<< V[1] <<" "<< Ysol[3] << endl;
        // the last line is useful to check the sovrapposion in a qualitative/intuitive way
    }
    fdata<<"\n"<<endl;
    
  }
  
  fdata.close();
  
  
  
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  #elif SESSION == 2 // ------------------------- CONVERGENCE: -------------------------
  // -----------------------------------------------------------------------------------
  
  double Y0[neq]={0., 0.001, 0., 0.};
  
  int N_min=100, N_max=10000, N_step=2;
  double t_max=10.;
  
  m2=m1; // solution with both equal to simplify
  L2=L1;
  
  
  ofstream fdata;         // declare Output stream class to operate on files
  fdata.open("dati_conv.dat"); // open output file
  
  
  cout<<"\t ##### SESSION 2 - convergence #####\n"<<endl;
  
  // RK methods ---------------------
  for (int j=1; j<=2; j++){
    
    for (int N=N_min; N<N_max; N=N*N_step){
      
      dt = t_max/(double)N;
      if (j==1) cout<<" --- dt="<<dt<<endl; // printing dt only the first time
      
      for (int i=0; i<neq; i++) Y[i]=Y0[i]; // Y must contein initial values when you start integrating
      t=0; // starting time    
      
      for (int i=0; i<N; i++){
        
        if (j==1) {
          RK4Step (t, Y, dYdt, dt, neq);
        } else {
          RK2Step (t, Y, dYdt, dt, neq);
        }
        t+=dt;
        
        for (int l=0; l<n_var; l++) NormAng(Y[l]); // this seems useless, but if I plot the graph I want thetas to stay in [ -pi, +pi ]
      }
      
      solution(t, Y0, Ysol);
      
      energy_coord(Y[0], Y[1], Y[2], Y[3], x1, x2, y1, y2, E);
      energy_coord(Ysol[0], Ysol[1], Ysol[2], Ysol[3], x1_sol, x2_sol, y1_sol, y2_sol, E_sol);
      
      fdata<< t <<" "<< dt <<" "<< fabs(E-E_sol) <<" "
        << fabs(Y[0]-Ysol[0]) <<" "<< fabs(Y[1]-Ysol[1]) <<" "<< fabs(Y[2]-Ysol[2]) <<" "<< fabs(Y[3]-Ysol[3]) << endl;
    
    }
    fdata<<"\n"<<endl;
    
  }
  
  
  
  // Verlet methods ---------------------
  for (int j=1; j<=3; j++){
    
    for (int N=N_min; N<N_max; N=N*N_step){
      
      dt = t_max/(double)N;
    
      for (int i=0; i<n_var; i++) {
        X[i]=Y0[i]; // X,V must contein initial values when you start integrating
        V[i]=Y0[i+n_var];
      }    
      t=0; // starting time
      
      for(int i=0; i<N; i++){ 
        
        if (j==1) {
          p_Verlet (X, V, acc_func, dt, n_var);
        } else {
          v_Verlet (X, V, acc_func, dt, n_var);
        }
        t+=dt;
        
        for (int l=0; l<n_var; l++) NormAng(X[l]); // this seems useless, but if I plot the graph I want thetas to stay in [ -pi, +pi ]
      }
      
      solution(t, Y0, Ysol);
      
      energy_coord(X[0], X[1], V[0], V[1], x1, x2, y1, y2, E);
      energy_coord(Ysol[0], Ysol[1], Ysol[2], Ysol[3], x1_sol, x2_sol, y1_sol, y2_sol, E_sol);
      
      fdata<< t <<" "<< dt <<" "<< fabs(E-E_sol) <<" "
        << fabs(X[0]-Ysol[0]) <<" "<< fabs(X[1]-Ysol[1]) <<" "<< fabs(V[0]-Ysol[2]) <<" "<< fabs(V[1]-Ysol[3]) << endl;
      
    }
    fdata<<"\n"<<endl;
    
  }
  
  fdata.close();
  
  
  
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  #elif SESSION == 3 // ----------------------- FLIP AND CHAOS: ------------------------
  // -----------------------------------------------------------------------------------
  
  int N_ang = 500 ; // 2 pi / N_ang = step for theta 1 and theta 2
  double fluc = 0.01, tol = 0.5;
  double theta_prec[2]={0., 0.}; // angle of the previous interaction
  
  bool flip = false, ok_flip = false; // control variables for the flip data
  bool rand = false, ok_rand = false; // control variables for the flip data
  
  double Ya[neq], Ya0[neq]={0., 0., 0., 0.}; // pendulum of reference
  double Yb[neq], Yb0[neq]={0., 0., 0., 0.}; // pendulum with fluctuation
  
  N=10*N;
  
  ofstream fdata_flip;         // declare Output stream class to operate on files
  fdata_flip.open("dati_flip.dat"); // open output file
  
  ofstream fdata_rand;         // declare Output stream class to operate on files
  fdata_rand.open("dati_rand.dat"); // open output file
  
  
  cout<<"\t ##### SESSION 3 - flip and chaos #####\n"<<endl;
  
  // loop over theta1 and theta2 ---------------------
  for (int a=0; a<N_ang; a++){
    Ya0[0] = (2*(double)a/(double)N_ang-1)*M_PI; // starts from -pi and ends in +pi
    Yb0[0] = Ya0[0] + fluc ;
  
    for (int b=0; b<N_ang; b++){
      Ya0[1] = (2*(double)b/(double)N_ang-1)*M_PI;
      Yb0[1] = Ya0[1] + fluc ;
      ok_flip = false;
      ok_rand = false;
      
      
      // -------------------------------------------------------------------------------
      
      // RK4 method ---------------------
      for (int i=0; i<neq; i++) { // Y must contein initial values when you start integrating
        Ya[i]=Ya0[i]; 
        Yb[i]=Yb0[i];
      }
      t=0; // starting time  
      k=0; // no flip or divergence
      
      // ------------------------------------------
      
      for (int i=0; i<N; i++){
        theta_prec[0] = Ya[0];
        theta_prec[1] = Ya[1];
        
        RK4Step (t, Ya, dYdt, dt, neq);
        RK4Step (t, Yb, dYdt, dt, neq);
        t+=dt;
        
        // check if the flip occurred
        if ( fabs(Ya[0]-M_PI) < 1. or fabs(Ya[1]-M_PI) < 1. ) { // only checking when the angle is close to pi
          if ( (Ya[0]-M_PI)*(theta_prec[0]-M_PI) < 0. or (Ya[1]-M_PI)*(theta_prec[1]-M_PI) < 0. ) { // one of mass flip
          //if ( (Ya[1]-M_PI)*(theta_prec[1]-M_PI) < 0. ) { // checking only when m2 flips, alternative
            flip = true; 
            k++;
          }
        }
        
        // check if it's diverging
        if ( fabs(Ya[0]-Yb[0])>tol or fabs(Ya[1]-Yb[1])>tol ) {
          rand = true;
          k++;
        }
        
        for (int l=0; l<n_var; l++) NormAng(Ya[l]); // this seems useless, but if I plot the graph I want thetas to stay in [ -pi, +pi ]
        for (int l=0; l<n_var; l++) NormAng(Yb[l]);
        
        // flip save ---------------------
        if (not ok_flip and flip) { // saves the first flip only
          fdata_flip << t <<" "<< Ya0[0] <<" "<< Ya0[1] << endl; // t, theta
          flip = false;
          ok_flip = true;
        }
        if (not ok_flip and i==N-1) fdata_flip << t <<" "<< Ya0[0] <<" "<< Ya0[1] << endl; // flip didn't occur
        
        // rand save ---------------------
        if (not ok_rand and rand) {
          fdata_rand << t <<" "<< Ya0[0] <<" "<< Ya0[1] << endl; // t, theta
          rand = false;
          ok_rand = true;
        }
        if (not ok_rand and i==N-1) fdata_rand << t <<" "<< Ya0[0] <<" "<< Ya0[1] << endl;
        
        // if both flip and divergence happened already, then skip to the next iteration to save time 
        if (k==2) continue;
        
      }
      
      // -------------------------------------------------------------------------------
    }
    fdata_flip<<endl;
    fdata_rand<<endl;
    
  }
  
  fdata_flip.close();
  fdata_rand.close();
  
  
  
  
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  #elif SESSION == 4 // ---------------------------- CHAOS: ----------------------------
  // -----------------------------------------------------------------------------------
  
  int N_ang = 200, N_tol=200; // 2 pi / N_ang = step for theta 1 and theta 2
  double fluc = 0.01, delta_mean[n_var]={0.,0.};
  
  
  double Ya[neq], Ya0[neq]={0., 0., 0., 0.}; // pendulum of reference
  double Yb[neq], Yb0[neq]={0., 0., 0., 0.}; // pendulum with fluctuation
  
  N=25*N;
  
  ofstream fdata_chaos;         // declare Output stream class to operate on files
  fdata_chaos.open("dati_chaos.dat"); // open output file
  
  
  cout<<"\t ##### SESSION 4 - chaos (fluctuation size) #####\n"<<endl;
  
  // loop over theta1 and theta2 ---------------------
  for (int a=0; a<N_ang; a++){
    Ya0[0] = (2*(double)a/(double)N_ang-1)*M_PI; // starts from -pi and ends in +pi
    Yb0[0] = Ya0[0] + fluc ;
  
    for (int b=0; b<N_ang; b++){
      Ya0[1] = (2*(double)b/(double)N_ang-1)*M_PI;
      Yb0[1] = Ya0[1] + fluc ;
      
      
      // -------------------------------------------------------------------------------
      
      // RK4 method ---------------------
      for (int i=0; i<neq; i++) { // Y must contein initial values when you start integrating
        Ya[i]=Ya0[i]; 
        Yb[i]=Yb0[i];
      }
      t=0; // starting time  
      k=0; // no flip or divergence
      
      // ------------------------------------------
      
      for (int i=0; i<N; i++){
        RK4Step (t, Ya, dYdt, dt, neq);
        RK4Step (t, Yb, dYdt, dt, neq);
        t+=dt;
        
        // check if it's diverging
        if (i>N-N_tol) { // 
          delta_mean[0] += fabs(Ya[0]-Yb[0]);
          delta_mean[1] += fabs(Ya[1]-Yb[1]);
        }
      }
      
      delta_mean[0] /= N_tol*M_PI;
      delta_mean[1] /= N_tol*M_PI;
      
      //fdata_chaos << delta_mean[0]+delta_mean[1])*0.5 <<" "<< Ya0[0] <<" "<< Ya0[1] << endl;
      fdata_chaos << Ya0[0] <<" "<< Ya0[1]  <<" "
      << delta_mean[0] <<" "<< delta_mean[1] <<" "<< (delta_mean[0]+delta_mean[1])*0.5 << " "
      << Ya[0] << " " << Ya[1] << " " << Ya[2] << " " << Ya[3] << " " 
      << Yb[0] << " " << Yb[1] << " " << Yb[2] << " " << Yb[3] << " " << endl;
      // -------------------------------------------------------------------------------
    }
    fdata_chaos<<endl;
  }
  fdata_chaos.close();
  
  
  #endif
}









// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------


// differential function for Runge-Kutta
void dYdt (double t, double *Y, double *R){
  // Compute the right-hand side of the ODE dy/dt = -t*y
  // Y contiene x0, y0, vx0, vy0
  double theta1 = Y[0], theta2 = Y[1]; // coord (angle)
  double d_theta1 = Y[2], d_theta2= Y[3]; // speed
  double g=9.81, L1=g_L1, m1=g_m1, L2=g_L2, m2=g_m2;
  
  R[0]=d_theta1; 
  R[1]=d_theta2;
  
  R[2] = ( // dd_theta1 (acc)
    -g*(2*m1+m2)*sin(theta1) 
    -m2*g*sin(theta1-2*theta2) 
    -2*sin(theta1-theta2)*m2*(
      d_theta2*d_theta2 *L2 
      +d_theta1*d_theta1 *L1*cos(theta1-theta2)
    ))/(
      L1*(2*m1+m2-m2*cos( 2*theta1-2*theta2))
    );
  
  R[3] = ( // dd_theta2
    2*sin(theta1-theta2)*( 
      d_theta1*d_theta1 *L1*(m1+m2) 
      +g*(m1+m2)*cos(theta1) 
      + d_theta2*d_theta2*L2*m2*cos(theta1-theta2)
    ))/(
      L2*(2*m1+m2-m2*cos( 2*theta1-2*theta2))
    );
    
}


// -------------------------------------------------------------------------------------
// differential function for Verlet
void acc_func(double *X, double *V, double *a){
  double theta1 = X[0], theta2 = X[1]; // coord (angle)
  double d_theta1 = V[0], d_theta2= V[1]; // speed
  double g=9.81, L1=g_L1, m1=g_m1, L2=g_L2, m2=g_m2;
  
  a[0] = ( // dd_theta1 (acc)
    -g*(2*m1+m2)*sin(theta1) 
    -m2*g*sin(theta1-2*theta2) 
    -2*sin(theta1-theta2)*m2*(
      d_theta2*d_theta2 *L2 
      +d_theta1*d_theta1 *L1*cos(theta1-theta2)
    ))/(
      L1*(2*m1+m2-m2*cos( 2*theta1-2*theta2))
    );
  
  a[1] = ( // dd_theta2
    2*sin(theta1-theta2)*( 
      d_theta1*d_theta1 *L1*(m1+m2) 
      +g*(m1+m2)*cos(theta1) 
      + d_theta2*d_theta2*L2*m2*cos(theta1-theta2)
    ))/(
      L2*(2*m1+m2-m2*cos( 2*theta1-2*theta2))
    );
}


// -------------------------------------------------------------------------------------
// analitic solution in small angles regime (supposing L1=L2 and m1=m2)
void solution(double t, double *Y0, double *Ysol){ 
  
  double g=9.81, L1=g_L1, m1=g_m1, L2=g_L2, m2=g_m2;
  
  double omega1 = sqrt((2-sqrt(2))*g/L1);
  double omega2 = sqrt((2+sqrt(2))*g/L1); 
  
  double A1 = (sqrt(2)*Y0[0] + Y0[1])*0.5;
  double A2 = (sqrt(2)*Y0[0] - Y0[1])*0.5;
  
  Ysol[0] =  ( A1*cos(omega1*t) + A2*cos(omega2*t) )/sqrt(2);
  Ysol[1] =    A1*cos(omega1*t) - A2*cos(omega2*t);
  
  Ysol[2] = - ( omega1*A1*sin(omega1*t) + omega2*A2*sin(omega2*t) )/sqrt(2);
  Ysol[3] = - ( omega1*A1*sin(omega1*t) - omega2*A2*sin(omega2*t) );
  
}


// -------------------------------------------------------------------------------------
// calculate the energy and the cartesian coordinates 
// recurrent operation: this will help with readability and lower the chance of error
void energy_coord(double theta1, double theta2, double omega1, double omega2, double &x1, double &x2, double &y1, double &y2, double &E){
  double g=9.81, L1=g_L1, m1=g_m1, L2=L1, m2=m1;
  double v1, v2;
  
  x1 =  L1*sin(theta1);
  y1 = -L1*cos(theta1);
  x2 = x1 +L2*sin(theta2);
  y2 = y1 -L2*cos(theta2);
  
  v1 = L1*omega1;
  v2 = L2*omega2;
  
  E = 0.5*((m1+m2)*v1*v1 + m2*v2*v2) +m2*cos(theta1-theta2)*v1*v2 +g*(m1*y1 + m2*y2);
}


// -------------------------------------------------------------------------------------
// normalize angles so they stay in [-pi, +pi]
void NormAng(double &ang){ 
  if (ang> M_PI) ang-= 2*M_PI;
  if (ang<-M_PI) ang+= 2*M_PI;
}


