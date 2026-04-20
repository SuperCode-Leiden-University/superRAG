reset    # force all graph-related options to default values

fname = "dati.dat"      # file name  

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
# fdata = { 1,     2,      3,      4,        5,        6, 7,  8,  9,  10 } =
#       = { tempo, theta1, theta2, d_theta1, d_theta2, E, x1, y1, x2, y2 }

# plots = { 1,      2,      3,      4,      5,       6,       7,            8       } =
#       = { theta1, theta2, omega1, omega2, phases1, phases2, delta energy, ev in space }


###################################### showing plots
array show_plot[8] = [ 1, 1, 1, 1, 1, 1, 1, 1 ]     # 1 = show, 0 = hide ###### 1, 1, 1, 1, 1, 1, 1, 1 ##### 0, 0, 0, 0, 0, 0, 0, 0 
#                  = [ θ1&2, ω1&2, Φ1&2, E, S ] 

save_graphs = 1     # 1 = salva i grafici, 0 = stampa a schermo

path = "project_04_graphs/"


###################################### methods
nb = 4  # numero dei blocchi, se si aumenta modificare anche il layout multiplot sotto!!!!!
array show_method[nb]; array method[nb]; array color1[nb]; array color2[nb]

show_method[1]=1; method[1] = "RK4    "; color1[1]="red";          color2[1]="goldenrod"
show_method[2]=1; method[2] = "RK2    "; color1[2]="dark-pink";    color2[2]="light-pink"
show_method[3]=1; method[3] = "pVerlet"; color1[3]="blue";         color2[3]="slategrey"
show_method[4]=1; method[4] = "vVerlet"; color1[4]="forest-green"; color2[4]="chartreuse"

#show_method[5]=1; method[5] = "..."; color1[5]="dark-violet";  color2[5]="purple"


###################################### animation & graphs parameters
ntail = 150  # number of points to draw in the tail
ninc = 80    # increment between frames
N = 0    # number of frames
pgif = 0  # pause between frames

graph_font = ",10"  # font/size di legenda ed etichette assi

L = 3 # L = L1+L2 = axis range [-L,+L]

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
if (show_plot[1] == 1) {
  set term x11 1 title "evolution in time, theta1(t)"
  set xlabel "t [s]"
  set ylabel "theta [rad]"
  
  
  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_theta1.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] {
    if (show_method[i]==0) { continue } #lines or linespoints
    plot fname using 1:2 index i-1 with lines title method[i] lt rgb color1[i] , \
         #cos(2.2*x)*pi/10. title "cosine function"
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph theta 1 saved"
  }
  
}



############################################################## THETA 2
if (show_plot[2] == 1) {
  set term x11 2 title "evolution in time, theta2(t)"
  set xlabel "t [s]"
  set ylabel "theta [rad]"
  
  
  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_theta2.png"
    stats fname nooutput
  }

  set multiplot layout row, col
  do for [i=1:nb:1] {
    if (show_method[i]==0) { continue } #lines or linespoints
    plot fname using 1:3 index i-1 with lines title method[i] lt rgb color2[i] , \
         #cos(2.2*x)*pi/10. title "cosine function"
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
if (show_plot[3] == 1) {
  set term x11 3 title "evolution in time, omega1(t)"
  set xlabel "t [s]"
  set ylabel "omega [rad/s]"

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_omega1.png"
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
    print " - graph omega 1 saved"
  }
}



############################################################## OMEGA 2
if (show_plot[4] == 1) {
  set term x11 4 title "evolution in time, omega2(t)"
  set xlabel "t [s]"
  set ylabel "omega [rad/s]"

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_omega2.png"
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
    print " - graph omega 2 saved"
  }
}


############################################################################################################################
############################################################## PHASES 1
if (show_plot[5] == 1) {
  set term x11 5 title "evolution of m1 in phase space"
  set xlabel "theta [rad]"
  set ylabel "omega [rad/s]"
  
  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_square,y_size_square
    set output path."graph_phases1.png"
    stats fname nooutput
  }
  
  set polar
  set grid polar
  
  set size ratio -1 # assi uguali
  #set size square # autoscale
  
  
  set multiplot layout row, col
  do for [i=1:nb:1] {
    if (show_method[i]==0) { continue }
    plot fname using 2:4 index i-1 with lines title method[i] lt rgb color1[i] #lines or linespoints
  }
  unset multiplot
  
  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph phases 1 saved"
  }
  unset polar
}



############################################################## PHASES 2
if (show_plot[6] == 1) {
  set term x11 6 title "evolution of m2 in phase space"
  set xlabel "theta [rad]"
  set ylabel "omega [rad/s]"

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_square,y_size_square
    set output path."graph_phases2.png"
    stats fname nooutput
  }
  
  set polar
  set grid polar
  
  set size ratio -1 # assi uguali
  #set size square # autoscale
  
  # optional labels for the angles:
  #R=3 # this should be set as the max value on the axis so the label is outside the graph
  #set_label(x, text) = sprintf("set label '%s' at (R*cos(%f)), (R*sin(%f))     center", text, x, x) # this places a label on the outside
  #eval set_label(0, "0")
  #eval set_label(pi*0.5, "90")
  #eval set_label(pi, "180")
  #eval set_label(pi*1.5, "270")
  
  
  set multiplot layout row, col
  do for [i=1:nb:1] {
    if (show_method[i]==0) { continue }
    plot fname using 3:5 index i-1 with lines title method[i] lt rgb color1[i] #lines or linespoints
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph phases 2 saved"
  }
  unset polar
}


reset    # force all graph-related options to default values
set autoscale xfixmin   # axis range automatically scaled to include the range  
set autoscale xfixmax   # of data to be plotted


############################################################################################################################
############################################################## ENERGY
if (show_plot[7] == 1) {
  set term x11 7 title "energy E(t)"
  set xlabel "t [s]"
  set ylabel "E [J]"

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_rectangle,y_size_rectangle
    set output path."graph_energy.png"
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
    print " - graph energy saved"
  }
}



############################################################################################################################
############################################################## EV IN SPACE
if (show_plot[8] == 1) {
  set term x11 8 title "evolution in space"
  set xlabel "x"
  set ylabel "y"

  if (save_graphs == 1) {
    # salva la gif:
    set terminal gif animate delay pgif enhanced font graph_font size x_size_square,y_size_square
    set output path."graph_ev_space.gif"
    stats fname nooutput
  }

  set size ratio -1 # assi uguali
  set xrange [-(L+0.5):(L+0.5)]
  set yrange [-(L+0.5):(L+1.) ]

  set object circle at first 0,0 size scr 0.005 fillcolor rgb 'black’ fillstyle solid  # center

  do for [ii=1:N:ninc] {
    im = ((ii - ntail) < 0 ? 1:ii-ntail) # first plot the point, then the tail
    
    set label 11 center at graph 0.5,char 1 sprintf("time = %1.3f", ii*0.001) font ",14"
    set bmargin 5
    
    plot fname using 7:8  index 0 every ::ii::ii notitle lt rgb color1[1], \
         fname using 7:8  index 0 every ::im::ii with lines title "m1, ".method[1] lt rgb color1[1], \
         fname using 9:10 index 0 every ::ii::ii notitle lt rgb color2[1], \
         fname using 9:10 index 0 every ::im::ii with lines title "m2, ".method[1] lt rgb color2[1], \
         fname using ($1-$1):($1-$1):7:8  index 0 every ::ii::ii notitle with vectors nohead filled lt rgb "dark-gray", \
         fname using 7:8:($9-$7):($10-$8) index 0 every ::ii::ii notitle with vectors nohead filled lt rgb "dark-gray", \
         \
         fname using 7:8  index 1 every ::ii::ii notitle lt rgb color1[2], \
         fname using 7:8  index 1 every ::im::ii with lines title "m1, ".method[2] lt rgb color1[2], \
         fname using 9:10 index 1 every ::ii::ii notitle lt rgb color2[2], \
         fname using 9:10 index 1 every ::im::ii with lines title "m2, ".method[2] lt rgb color2[2], \
         fname using ($1-$1):($1-$1):7:8  index 1 every ::ii::ii notitle with vectors nohead filled lt rgb "dark-gray", \
         fname using 7:8:($9-$7):($10-$8) index 1 every ::ii::ii notitle with vectors nohead filled lt rgb "dark-gray", \
         \
         fname using 7:8  index 2 every ::ii::ii notitle lt rgb color1[3], \
         fname using 7:8  index 2 every ::im::ii with lines title "m1, ".method[3] lt rgb color1[3], \
         fname using 9:10 index 2 every ::ii::ii notitle lt rgb color2[3], \
         fname using 9:10 index 2 every ::im::ii with lines title "m2, ".method[3] lt rgb color2[3], \
         fname using ($1-$1):($1-$1):7:8  index 2 every ::ii::ii notitle with vectors nohead filled lt rgb "dark-gray", \
         fname using 7:8:($9-$7):($10-$8) index 2 every ::ii::ii notitle with vectors nohead filled lt rgb "dark-gray", \
         \
         fname using 7:8  index 3 every ::ii::ii notitle lt rgb color1[4], \
         fname using 7:8  index 3 every ::im::ii with lines title "m1, ".method[4] lt rgb color1[4], \
         fname using 9:10 index 3 every ::ii::ii notitle lt rgb color2[4], \
         fname using 9:10 index 3 every ::im::ii with lines title "m2, ".method[4] lt rgb color2[4], \
         fname using ($1-$1):($1-$1):7:8  index 3 every ::ii::ii notitle with vectors nohead filled lt rgb "dark-gray", \
         fname using 7:8:($9-$7):($10-$8) index 3 every ::ii::ii notitle with vectors nohead filled lt rgb "dark-gray", \
         
    unset label
  }
  
  ########################

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_square+100,y_size_square+100
    set output path."graph_ev_space.png"
    stats fname nooutput
  } else { clear }

  set multiplot layout row, col
  do for [i=1:nb:1] {
    if (show_method[i]==0) { continue }
    plot fname using 7:8  index i-1 with lines title "m1, ".method[i] lt rgb color1[i],  fname using 9:10 index i-1 with lines title "m2, ".method[i] lt rgb color2[i]
  }
  unset multiplot

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graphs space saved"
  }
}








