reset    # force all graph-related options to default values

fname = "dati_small.dat"      # file name  

set autoscale xfixmin   # axis range automatically scaled to include the range  
set autoscale xfixmax   # of data to be plotted

set tics font   ",10"

#set logscale x 10 # scala logaritmica in base 10
#set logscale y 10 # scala logaritmica in base 10

#set lmargin at screen 0.1    # set size of left margin
#set rmargin at screen 0.82   # set size of right margin
#set bmargin at screen 0.12   # set size of bottom margin
#set tmargin at screen 0.95   # set size of top margin


############################################################################################################################

# struttura dei dati: prima RK4, poi Verlet

# fdata = { 1,     2,  3,    4,          5,          6,          7          } =
#       = { tempo, dt, E-E0, theta1-sol, theta2-sol, omega1-sol, omega2-sol }

# plots = { 0,           1,      2,      3,      4,      5,     } =
#       = { convergence, theta1, theta2, omega1, omega2, energy }


###################################### showing plots
N_plots = 5
array show_plot[N_plots] = [ 1, 1, 1, 1, 1 ]     # 1 = show, 0 = hide ###### 1, 1, 1, 1, 1 ##### 0, 0, 0, 0, 0
#                        = [ θ1&2, ω1&2, E ] 

both        = 0   # 1 = analitic and numeric, 0 = difference
save_graphs = 1   # 1 = salva i grafici,      0 = stampa a schermo

path = "project_04_graphs_small/"


###################################### methods
nb = 4  # numero dei blocchi, se si aumenta modificare anche il layout multiplot sotto!!!!!
array show_method[nb]; array method[nb]; array color1[nb]; array color2[nb]

show_method[1]=1; method[1] = "RK4    "; color1[1]="red";          color2[1]="goldenrod"
show_method[2]=1; method[2] = "RK2    "; color1[2]="dark-pink";    color2[2]="light-pink"
show_method[3]=1; method[3] = "pVerlet"; color1[3]="blue";         color2[3]="slategrey"
show_method[4]=1; method[4] = "vVerlet"; color1[4]="forest-green"; color2[4]="chartreuse"

#show_method[5]=1; method[5] = "..."; color1[5]="dark-violet";  color2[5]="purple"



###################################### graphs parameters
graph_font = ",10"  # font/size di legenda ed etichette assi

x_size_rectangle = 900; y_size_rectangle = 400  # dim in pixel dell'immagine
x_size_square = 500; y_size_square = x_size_square  # dim in pixel dell'immagine


###################################### layout plot (row, col) ### automatic 
show_method_tot =0
do for [i=1:nb:1] { show_method_tot = show_method_tot + show_method[i] }

if (show_method_tot<=2) { 
  row=1; col=show_method_tot 
} 
if (show_method_tot>2 && show_method_tot<=4) { 
  row=2; col=2 
}
if (show_method_tot>4 && show_method_tot<=6) { 
  row=3; col=2 
}

#set size ratio -1 # assi uguali
#set size square # autoscale



############################################################################################################################
#### ------------------------------------------------------------------------------------------------------------------- ###
#### ----------------------------------------------------- GRAFICI ----------------------------------------------------- ###
#### ------------------------------------------------------------------------------------------------------------------- ###
############################################################################################################################

############################################################## THETA 1
if (show_plot[1] == 1 && both==0) {
  set term x11 1 title "error of theta1(t)"
  set xlabel "t [s]"
  set ylabel "theta_1 err [rad]"
  
  
  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_delta_theta1.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] {
    if (show_method[i]==0) { continue }
    plot fname using 1:4 index i-1 with lines title method[i] lt rgb color1[i] #lines or linespoints
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph theta 1 saved"
  }
  
}



############################################################## THETA 2
if (show_plot[2] == 1 && both==0) {
  set term x11 2 title "error of theta2(t)"
  set xlabel "t [s]"
  set ylabel "theta_2 err [rad]"
  
  
  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_delta_theta2.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] {
    if (show_method[i]==0) { continue }
    plot fname using 1:5 index i-1 with lines title method[i] lt rgb color2[i] #lines or linespoints
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph theta 2 saved"
  }
  
}



############################################################################################################################
############################################################## OMEGA 1
if (show_plot[3] == 1 && both==0) {
  set term x11 3 title "error of omega1(t)"
  set xlabel "t [s]"
  set ylabel "omega_1 err [rad/s]"

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_delta_omega1.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] {
    if (show_method[i]==0) { continue }
    plot fname using 1:6 index i-1 with lines title method[i] lt rgb color1[i] #lines or linespoints
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph omega 1 saved"
  }
}



############################################################## OMEGA 2
if (show_plot[4] == 1 && both==0) {
  set term x11 4 title "error of omega2(t)"
  set xlabel "t [s]"
  set ylabel "omega_2 err [rad/s]"

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_delta_omega2.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] {
    if (show_method[i]==0) { continue }
    plot fname using 1:7 index i-1 with lines title method[i] lt rgb color2[i] #lines or linespoints
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph omega 2 saved"
  }
}


############################################################################################################################
############################################################## ENERGY
if (show_plot[5] == 1 && both==0) {
  set term x11 5 title "energy E(t)"
  set xlabel "t [s]"
  set ylabel "E err [J]"

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_delta_energy.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] {
    if (show_method[i]==0) { continue }
    plot fname using 1:3 index i-1 with lines title method[i] lt rgb color1[i] #lines or linespoints
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph energy saved"
  }
}




############################################################################################################################
############################################################################################################################
############################################################################################################################



############################################################## THETA 1
if (show_plot[1] == 1 && both==1) {
  set term x11 1 title "error of theta1(t)"
  set xlabel "t [s]"
  set ylabel "theta_1 err [rad]"
  
  
  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_theta1.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] { #lines or linespoints
    if (show_method[i]==0) { continue }
    plot fname using 1:10 index i-1 with linespoints title method[i] lt rgb color1[i], \
         fname using 1:11 index i-1 with lines title "sol" lt rgb "black"
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph theta 1 saved"
  }
  
}



############################################################## THETA 2
if (show_plot[2] == 1 && both==1) {
  set term x11 2 title "error of theta2(t)"
  set xlabel "t [s]"
  set ylabel "theta_2 err [rad]"
  
  
  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_theta2.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] { #lines or linespoints
    if (show_method[i]==0) { continue }
    plot fname using 1:12 index i-1 with linespoints title method[i] lt rgb color2[i], \
         fname using 1:13 index i-1 with lines title "sol" lt rgb "black"
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph theta 2 saved"
  }
  
}



############################################################################################################################
############################################################## OMEGA 1
if (show_plot[3] == 1 && both==1) {
  set term x11 3 title "error of omega1(t)"
  set xlabel "t [s]"
  set ylabel "omega_1 err [rad/s]"

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_omega1.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] { #lines or linespoints
    if (show_method[i]==0) { continue }
    plot fname using 1:14 index i-1 with linespoints title method[i] lt rgb color1[i], \
         fname using 1:15 index i-1 with lines title "sol" lt rgb "black"
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph omega 1 saved"
  }
}



############################################################## OMEGA 2
if (show_plot[4] == 1 && both==1) {
  set term x11 4 title "error of omega2(t)"
  set xlabel "t [s]"
  set ylabel "omega_2 err [rad/s]"

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_omega2.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] { #lines or linespoints
    if (show_method[i]==0) { continue }
    plot fname using 1:16 index i-1 with linespoints title method[i] lt rgb color2[i], \
         fname using 1:17 index i-1 with lines title "sol" lt rgb "black"
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph omega 2 saved"
  }
}


############################################################################################################################
############################################################## ENERGY
if (show_plot[5] == 1 && both==1) {
  set term x11 5 title "energy E(t)"
  set xlabel "t [s]"
  set ylabel "E err [J]"

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_energy.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] { #lines or linespoints
    if (show_method[i]==0) { continue }
    plot fname using 1:8 index i-1 with linespoints title method[i] lt rgb color1[i], \
         fname using 1:9 index i-1 with lines title "sol" lt rgb "black"
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph energy saved"
  }
}








