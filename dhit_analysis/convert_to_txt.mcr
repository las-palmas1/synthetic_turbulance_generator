#!MC 1410
$!VarSet |MFBD| = 'C:\Program Files\Tecplot\Tecplot 360 EX 2016 R2'
$!READDATASET  '"C:\Users\User\Documents\tasks\computitions_and post_processing\dhit\64_cells\data_for_analysis\lazurit\first_step_new_grid\bin\TEC_FLOW_T-1_B-1.plt"'
$!WRITEDATASET  'C:\Users\User\Documents\tasks\computitions_and post_processing\dhit\64_cells\data_for_analysis\lazurit\first_step_new_grid\txt\data_0.dat' 
  INCLUDETEXT = NO
  INCLUDEGEOM = NO
  INCLUDEDATASHARELINKAGE = YES
  BINARY = FALSE
  USEPOINTFORMAT = YES
  PRECISION = 9
  TECPLOTVERSIONTOWRITE = TECPLOTCURRENT
$!RemoveVar |MFBD|
$!Quit
