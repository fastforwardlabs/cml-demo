
#Run this only if you are not using a custom docker image
#To build a custom docker image for this project run commands like in build-engine.sh

##NOTE: You need a session with at least 8GB memory to run this

cd
!cp /home/cdsw/utils/cdsw-build.sh .
!chmod 755 /home/cdsw/cdsw-build.sh
!sh /home/cdsw/cdsw-build.sh
!if [[ ! -d /home/cdsw/data ]]; then mkdir -m 755 /home/cdsw/data; fi
!Rscript /home/cdsw/utils/install.R
