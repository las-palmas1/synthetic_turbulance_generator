#!MC 1410
$!VarSet |MFBD| = 'C:\Program Files\Tecplot\Tecplot 360 EX 2016 R2'
$!READDATASET  '"StandardSyntax" "1.0" "FEALoaderVersion" "435" "FILENAME_File" "C:\Users\User\Documents\tasks\computitions_and post_processing\diht\32_cells\data_for_analysis\cfx\bin\280.trn" "AutoAssignStrandIDs" "Yes"' DATASETREADER = 'ANSYS CFX (FEA)'
$!WRITEDATASET  'C:\Users\User\Documents\tasks\computitions_and post_processing\diht\32_cells\data_for_analysis\cfx\txt\cfx_0.dat' 
  INCLUDETEXT = NO
  INCLUDEGEOM = NO
  INCLUDEDATASHARELINKAGE = YES
  ZONELIST = [1]
  VARPOSITIONLIST =  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
  BINARY = FALSE
  USEPOINTFORMAT = YES
  PRECISION = 9
  TECPLOTVERSIONTOWRITE = TECPLOTCURRENT
$!READDATASET  '"StandardSyntax" "1.0" "FEALoaderVersion" "435" "FILENAME_File" "C:\Users\User\Documents\tasks\computitions_and post_processing\diht\32_cells\data_for_analysis\cfx\bin\main_001.res" "AutoAssignStrandIDs" "Yes"' DATASETREADER = 'ANSYS CFX (FEA)'
$!WRITEDATASET  'C:\Users\User\Documents\tasks\computitions_and post_processing\diht\32_cells\data_for_analysis\cfx\txt\cfx_1.dat' 
  INCLUDETEXT = NO
  INCLUDEGEOM = NO
  INCLUDEDATASHARELINKAGE = YES
  ZONELIST = [1]
  VARPOSITIONLIST =  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
  BINARY = FALSE
  USEPOINTFORMAT = YES
  PRECISION = 9
  TECPLOTVERSIONTOWRITE = TECPLOTCURRENT
$!RemoveVar |MFBD|
$!Quit
