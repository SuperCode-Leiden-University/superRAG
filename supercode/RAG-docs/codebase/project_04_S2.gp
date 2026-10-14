reset    # force all graph-related options to default values

fname = "dati_conv.dat"      # file name 

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

#set size ratio -1 # assi uguali
#set size square # autoscale


############################################################################################################################
#### ------------------------------------------------------------------------------------------------------------------- ###
#### ----------------------------------------------------- GRAFICI ----------------------------------------------------- ###
#### ------------------------------------------------------------------------------------------------------------------- ###
############################################################################################################################


############################################################## CONVERGENCE

#set size square # assi uguali

set logscale x 10 # scala logaritmica in base 10
set logscale y 10 # scala logaritmica in base 10

set term x11 0 title "convergence"
set xlabel "dt [s]"
set ylabel "theta_1 err [rad]"
set key left top # legenda 

set format x "10^{%T}"
set format y "10^{%T}"


if (save_graphs == 1) {
  # salva il grafico:
  set terminal png enhanced font graph_font size x_size_square,y_size_square
  set output path."graph_convergence.png"
  stats fname nooutput
}

#set multiplot layout row, col
#do for [i=3:5:1] {
  i=4
  f4(x) = (x*x*x*x)/5. + 5e-09
  f2(x) = (x*x)/10. + 1e-07
  #points, lines or linespoints
  plot fname using 2:i index 0 with linespoints title method[1] lt rgb color1[1] , \
       fname using 2:i index 1 with linespoints title method[2] lt rgb color1[2] , \
       fname using 2:i index 2 with linespoints title method[3] lt rgb color1[3] , \
       fname using 2:i index 3 with linespoints title method[4] lt rgb color1[4] , \
       f4(x) title "x*x*x*x" with lines lt rgb "black", \
       f2(x) title "x*x" with lines lt rgb "dark-gray"
#}
#unset multiplot

if (save_graphs == 1) {
  set terminal wxt # output a schermo
  set output
  print " - graph convergence saved"
}









