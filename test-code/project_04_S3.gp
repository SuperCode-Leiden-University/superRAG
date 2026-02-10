reset    # force all graph-related options to default values

fname_flip = "dati_flip.dat"  # file name  
fname_rand = "dati_rand.dat" 

set autoscale xfixmin         # axis range automatically scaled to include the range  
set autoscale xfixmax         # of data to be plotted

set tics font   ",10"

#set logscale x 10 # scala logaritmica in base 10
#set logscale y 10 # scala logaritmica in base 10

#set lmargin at screen 0.1    # set size of left margin
#set rmargin at screen 0.82   # set size of right margin
#set bmargin at screen 0.12   # set size of bottom margin
#set tmargin at screen 0.95   # set size of top margin


############################################################################################################################

# struttura dei dati: 
# fdata = { 1,     2,      3,     } =
#       = { tempo, theta1, theta2 }

# plots = { 1,    2    } =
#       = { flip, rand }


###################################### showing plots
array show_plot[2] = [ 1, 1 ]     # 1 = show, 0 = hide
#                  = [ f, r ] 

save_graphs = 1     # 1 = salva i grafici, 0 = stampa a schermo

path = "project_04_graphs/"



###################################### graphs parameters

graph_font = ",10"  # font/size di legenda ed etichette assi

x_size_rectangle = 900; y_size_rectangle = 400  # dim in pixel dell'immagine
x_size_square = 500; y_size_square = x_size_square  # dim in pixel dell'immagine





############################################################################################################################
#### ------------------------------------------------------------------------------------------------------------------- ###
#### ----------------------------------------------------- GRAFICI ----------------------------------------------------- ###
#### ------------------------------------------------------------------------------------------------------------------- ###
############################################################################################################################

set size ratio -1 # assi uguali
#set size square # autoscale

############################################################## FLIP
if (show_plot[1] == 1) {
  set term x11 1 title "flip"
  set xlabel "theta1 [rad]"
  set ylabel "theta2 [rad]"
  
  set cblabel "time [s]"
  #set cbrange [0.001:200]
  #set logscale cb
  
  #set label "time [s]" at 3.4,-3.5 tc rgb "black" font graph_font front
  #set bmargin 5
  #show label
  
  set xrange [-pi:pi]
  set yrange [-pi:pi]

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_square,y_size_square
    set output path."graph_flip.png"
    stats fname_flip nooutput
  }
  
  set pm3d map # color map
  splot fname_flip using 2:3:1 index 0 notitle
  

  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph flip saved"
  }
}



############################################################## RANDOM
if (show_plot[2] == 1) {
  set term x11 2 title "divergence of trajectory"
  set xlabel "theta1 [rad]"
  set ylabel "theta2 [rad]"
  
  set cblabel "time [s]"
  #set cbrange [0.001:200]
  #set logscale cb

  if (save_graphs == 1) {
    # salva il grafico:
    set terminal png enhanced font graph_font size x_size_square,y_size_square
    set output path."graph_random.png"
    stats fname_rand nooutput
  }
  
  set pm3d map # color map
  splot fname_rand using 2:3:1 index 0 notitle 
  
  if (save_graphs == 1) {
    set terminal wxt # output a schermo
    set output
    print " - graph random saved"
  }
}



############################################################################################################################







