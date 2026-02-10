using namespace std;

// ------------------- Prototypes

void EulerStep (double, double *,   void (*)(double,  double *, double *), double, int);
void RK2Step   (double, double *,   void (*)(double,  double *, double *), double, int);
void RK4Step   (double, double *,   void (*)(double,  double *, double *), double, int);
void v_Verlet  (double *, double *, void (*)(double*, double *, double *), double, int);
void p_Verlet  (double *, double *, void (*)(double*, double *, double *), double, int);


void mean_Verlet  (double *, double *, void (*)(double*, double *, double *), double, int);





// ---------------------------------------------------------------------------------------
// ------------------------------------ Functions ----------------------------------------
// ---------------------------------------------------------------------------------------


// -------------------- Euler method
void EulerStep (double t, double *Y, void (*RHS_Func)(double, double *, double *), double dt, int neq){
  // Take one step dt using Euler method for the solution of dY/dt = rhs.
  // Here neq is the number of ODE (the dimensionality of Y[]) and *RHSFunc() is a
  // pointer to the actual function (in this case it points to dYdt()) that
  // calculates the right-hand side of the system of equations.
  
  double rhs[neq];
  RHS_Func (t, Y, rhs); // RHS_Func contains the differential equations of the problem 
  for (int k = 0; k < neq; k++) {
    Y[k] += dt*rhs[k]; // Update solution array
  }
}


// -------------------- RK2 method
void RK2Step(double t, double *Y, void (*RHS_Func)(double, double *, double *), double dt, int neq){ 
  double Y1[neq], k1[neq], k2[neq]; 
  
  RHS_Func(t, Y, k1);
  for (int k = 0; k < neq; k++) {
    Y1[k] = Y[k]+0.5*dt*k1[k];
  }
  
  RHS_Func(t+0.5*dt,Y1,k2);
  for (int k = 0; k < neq; k++) {
    Y[k] += dt*k2[k];
  }
} 


// -------------------- RK4 method
void RK4Step(double t, double *Y, void (*RHS_Func)(double, double *, double *), double dt, int neq){ 
  double Y1[neq], Y2[neq], Y3[neq];
  double k1[neq], k2[neq], k3[neq], k4[neq]; 
  
  RHS_Func(t, Y, k1);
  for (int k = 0; k < neq; k++) {
    Y1[k] = Y[k]+0.5*dt*k1[k];
  }
  
  RHS_Func(t+0.5*dt, Y1, k2);
  for (int k = 0; k < neq; k++) {
    Y2[k] = Y[k]+0.5*dt*k2[k];
  }
  
  RHS_Func(t+0.5*dt, Y2, k3);
  for (int k = 0; k < neq; k++) {
    Y3[k] = Y[k]+dt*k3[k];
  }
  
  RHS_Func(t+dt, Y3, k4);
  for (int k = 0; k < neq; k++) {
    Y[k] += ( k1[k] +2*k2[k] +2*k3[k] +k4[k] )*dt/6;
  }
}


// -------------------- velocity-Verlet Method
void v_Verlet (double *x, double *v, void (*acc)(double *, double *, double*), double dt, int n_var){
  double a[n_var];
  
  acc(x,v,a);
  for(int k=0; k<n_var; k++){
    v[k]+=0.5*dt*a[k];
  }
  for(int k=0; k<n_var; k++){
    x[k]+=dt*v[k];
  }
  acc(x,v,a);
  for(int k=0; k<n_var; k++){
    v[k]+=0.5*dt*a[k];
  }
}


// -------------------- position-Verlet Method
void p_Verlet (double *x, double *v, void (*acc)(double *, double *, double*), double dt, int n_var){
  double a[n_var];
  
  for(int k=0; k<n_var; k++){
    x[k]+=0.5*dt*v[k];
  }
  
  acc(x,v,a);
  for(int k=0; k<n_var; k++){
    v[k]+=dt*a[k];
  }
  for(int k=0; k<n_var; k++){
    x[k]+=0.5*dt*v[k];
  }
  
}









