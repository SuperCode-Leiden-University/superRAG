reset    # force all graph-related options to default values

fname_chaos = "dati_chaos.dat" 

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
#       = { delta, theta1, theta2 }


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


############################################################################################################################
############################################################## CHAOS DELTA TRAJECTORY

set term x11 0 title "divergence of trajectory"
set xlabel "theta_1 [rad]"
set ylabel "theta_2 [rad]"

set cblabel "delta phase trajectory"
set cbrange [0.3:]
set logscale cb

set xrange [-pi:pi]
set yrange [-pi:pi]

if (save_graphs == 1) {
  # salva il grafico:
  set terminal png enhanced font graph_font size x_size_square,y_size_square
  set output path."graph_chaos_df.png"
  stats fname_chaos nooutput
}
  
set pm3d map # color map
splot fname_chaos using 1:2:( sqrt( ($6-$10)**2 + ($7-$11)**2 + ($8-$12)**2 + ($9-$13)**2 ) ) index 0 notitle 

if (save_graphs == 1) {
  set terminal wxt # output a schermo
  set output
  print " - graph chaos df saved"
}



############################################################## CHAOS DELTA THETA

set term x11 2 title "divergence of theta"
set xlabel "theta_1 [rad]"
set ylabel "theta_2 [rad]"

set cblabel "delta theta at final t"
set cbrange [0.2:160]
set logscale cb

set xrange [-pi:pi]
set yrange [-pi:pi]

if (save_graphs == 1) {
  # salva il grafico:
  set terminal png enhanced font graph_font size x_size_square,y_size_square
  set output path."graph_chaos_dtheta.png"
  stats fname_chaos nooutput
}
  
set pm3d map # color map
#splot fname_chaos using 1:2:(sqrt( ($6-$10)**2 + ($7-$11)**2 )) index 0 notitle 
#splot fname_chaos using 1:2:( (abs($6-$10) + abs($7-$11))*0.5 ) index 0 notitle 
splot fname_chaos using 1:2:( abs($7-$11) ) index 0 notitle 

if (save_graphs == 1) {
  set terminal wxt # output a schermo
  set output
  print " - graph chaos dt saved"
}





