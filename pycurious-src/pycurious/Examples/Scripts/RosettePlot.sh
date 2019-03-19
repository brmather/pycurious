#!/bin/bash

oFile='RosetteTest.ps'

# Several plots... 
# plot of theta vs CPD

J=' -JX8c/6c'
R=' -R0/180/-35/-15'
psbasemap $J $R -Ba30/a10g5::neWS -K > $oFile
cat iap_rosette.txt | psxy -R -J -O -K -W2p,blue>> $oFile
echo '90 -35 Azimuth (@+o@+)' | pstext -R -J -O -K -N -Ya-1.25c >> $oFile
echo '0 -25 CPD (km)' | pstext -R -J -O -K -N -F+a90 -Xa-1.25c >> $oFile

psbasemap $J -R0/2/-35/-15 -B/a10g5::neWS -O -K -Y10c >> $oFile
echo '1 -23.7 -19.7 -22.3 -24.48 -29.09' | psxy -R -J -EY -G127 -O -K >> $oFile
echo '1 -23.7 -19.7 -22.3 -24.48 -29.09' | psxy -R -J -Sc0.2c -G127 -O -K >> $oFile

# box & whisker
# fancy rosette...?